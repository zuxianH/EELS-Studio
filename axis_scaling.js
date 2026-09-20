// Shift-drag an axis to zoom about its midpoint. Plain drags stay with Plotly.
export default function ({ data }) {
    const selector = data?.selector ?? ".st-key-spectrum_axis_target .js-plotly-plot";
    // Streamlit may replace the whole Plotly node when the spectrum changes.
    // Keep the user's viewport in this browser tab, across component remounts.
    const viewKey = Symbol.for(data?.viewport_key ?? "eels.spectrum.viewport");
    const views = window[viewKey] ??= {};
    const axisNames = (layout) => Object.keys(layout ?? {}).filter(name => /^[xy]axis\d*$/.test(name));
    const axisRevision = (layout, axis) => layout[axis]?.uirevision ?? layout.uirevision;
    let chart = null;
    let lastRender = null;
    let restoring = false;
    const rememberView = (event) => {
        if (restoring || !chart?._fullLayout) return;
        if (!Object.keys(event).some(key => /^[xy]axis\d*\.(autorange|range)/.test(key))) return;
        // A relayout can also move matched axes without listing them in its payload.
        // Remember their actual ranges together so restoring never unlinks subplots.
        for (const axis of axisNames(chart._fullLayout)) {
            const layout = chart._fullLayout[axis];
            if (layout.autorange) {
                delete views[axis];
            } else if (layout.range?.every(Number.isFinite)) {
                views[axis] = { range: [...layout.range], revision: axisRevision(chart._fullLayout, axis) };
            }
        }
    };
    const restoreView = () => {
        const layout = chart?._fullLayout;
        const revision = layout?.meta?.eels_render_revision;
        if (revision === undefined || revision === lastRender || restoring) return;
        lastRender = revision;
        const update = {};
        for (const axis of axisNames(layout)) {
            const saved = views[axis];
            if (!saved) continue;
            // Numeric limit edits intentionally replace the saved view of that axis.
            if (saved.revision !== axisRevision(layout, axis)) {
                delete views[axis];
                continue;
            }
            update[`${axis}.range`] = [...saved.range];
            update[`${axis}.autorange`] = false;
        }
        if (Object.keys(update).length && window.Plotly?.relayout) {
            restoring = true;
            Promise.resolve(window.Plotly.relayout(chart, update)).finally(() => {
                restoring = false;
            });
        }
    };
    const connect = () => {
        const next = document.querySelector(selector);
        // Plotly.purge removes emitter methods before Streamlit detaches the node.
        if (chart) {
            chart.removeListener?.("plotly_relayout", rememberView);
            chart.removeListener?.("plotly_afterplot", restoreView);
        }
        if (next !== chart) lastRender = null;
        chart = next && typeof next.on === "function" ? next : null;
        if (chart) {
            chart.on("plotly_relayout", rememberView);
            chart.on("plotly_afterplot", restoreView);
            restoreView();
        }
    };
    const observer = new MutationObserver(connect);
    observer.observe(document.body, { childList: true, subtree: true });
    const prepareInteraction = (event) => {
        if (event.target.closest?.(selector)) connect();
    };
    document.addEventListener("pointerdown", prepareInteraction, true);
    connect();

    let drag = null;
    let pending = null;
    let frame = null;
    let updating = false;
    let disposed = false;

    const flush = async () => {
        frame = null;
        if (disposed || updating || !pending) return;
        const { chart, update } = pending;
        pending = null;
        if (!chart.isConnected) return;
        updating = true;
        try {
            await window.Plotly.relayout(chart, update);
        } finally {
            updating = false;
            if (!disposed && pending) frame = requestAnimationFrame(flush);
        }
    };
    const move = (event) => {
        if (!drag) return;
        if (!drag.chart.isConnected) {
            drag = null;
            return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
        const distance = drag.horizontal
            ? event.clientX - drag.x : drag.y - event.clientY;
        // One axis-length of movement changes the span by a factor of four.
        const factor = Math.exp(Math.max(-8, Math.min(8, -distance / drag.length * Math.log(4))));
        const range = [drag.midpoint - drag.halfSpan * factor,
                       drag.midpoint + drag.halfSpan * factor];
        if (!range.every(Number.isFinite) || range[0] === range[1]) return;
        pending = { chart: drag.chart, update: {
            [`${drag.axis}.range`]: range,
            [`${drag.axis}.autorange`]: false,
        } };
        if (frame === null && !updating) frame = requestAnimationFrame(flush);
    };
    const start = (event) => {
        if (event.button !== 0 || !event.shiftKey) return;
        const chart = event.target.closest?.(selector);
        if (!chart || typeof window.Plotly?.relayout !== "function") return;
        const target = event.target.closest(".ewdrag, .wdrag, .edrag, .nsdrag, .ndrag, .sdrag");
        if (!target || !chart.contains(target)) return;
        // Plotly groups each subplot's drag handles under its axis pair (xy,
        // x2y2, ...). Resolve the handle's own axis, including the lower panel.
        const subplot = target.closest("g")?.getAttribute("class")?.match(/(?:^|\s)(x\d*)(y\d*)(?:\s|$)/);
        if (!subplot) return;
        const horizontal = target.matches(".ewdrag, .wdrag, .edrag");
        const axisId = subplot[horizontal ? 1 : 2];
        const axis = `${axisId[0]}axis${axisId.slice(1)}`;
        const layout = chart._fullLayout?.[axis];
        if (!layout || layout.fixedrange || !["linear", "log"].includes(layout.type)) return;
        const [low, high] = layout.range;
        if (![low, high].every(Number.isFinite) || low === high || !(layout._length > 0)) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        drag = { chart, axis, horizontal, x: event.clientX, y: event.clientY,
                 length: layout._length, midpoint: low / 2 + high / 2,
                 halfSpan: high / 2 - low / 2 };
    };
    const end = (event) => {
        if (!drag) return;
        move(event);
        drag = null;
    };
    const blur = () => { drag = null; };
    // Plotly's own scrollZoom listener occasionally misses a preventDefault during a
    // fast scroll burst, letting the delta fall through to the page. Cancel the
    // browser's default scroll for every wheel event over the chart, in the capture
    // phase, so the page never scrolls no matter how Plotly's handler keeps up;
    // propagation continues so Plotly still sees the event and zooms normally.
    const wheel = (event) => {
        if (event.target.closest?.(selector)) event.preventDefault();
    };
    // Delegation survives Streamlit replacing the chart on a rerun.
    document.addEventListener("mousedown", start, true);
    document.addEventListener("mousemove", move, true);
    document.addEventListener("mouseup", end, true);
    document.addEventListener("wheel", wheel, true);
    window.addEventListener("blur", blur);
    return () => {
        disposed = true;
        observer.disconnect();
        document.removeEventListener("pointerdown", prepareInteraction, true);
        // Plotly.purge removes emitter methods before Streamlit detaches the node.
        if (chart) {
            chart.removeListener?.("plotly_relayout", rememberView);
            chart.removeListener?.("plotly_afterplot", restoreView);
        }
        drag = pending = null;
        if (frame !== null) cancelAnimationFrame(frame);
        document.removeEventListener("mousedown", start, true);
        document.removeEventListener("mousemove", move, true);
        document.removeEventListener("mouseup", end, true);
        document.removeEventListener("wheel", wheel, true);
        window.removeEventListener("blur", blur);
    };
}
