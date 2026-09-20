// Relay Plotly's drawn/edited rectangle to Python. Heatmaps do not emit
// point-selection events for every pixel, so use shape relayout events.
export default function ({ data, setTriggerValue }) {
    let chart = null;
    const onRelayout = (event) => {
        if (!Object.keys(event).some(key => /^shapes(?:\[|$)/.test(key))) return;
        const shapes = event.shapes ?? chart?.layout?.shapes ?? [];
        const rectangle = [...shapes].reverse().find(shape => shape.type === "rect");
        if (!rectangle) return;
        const { x0, x1, y0, y1 } = rectangle;
        if (![x0, x1, y0, y1].every(Number.isFinite)) return;
        setTriggerValue("selected", {
            x0, x1, y0, y1,
            preview_id: data.preview_id,
            event_id: crypto.randomUUID(),
        });
    };
    const selector = ".st-key-angle_rectangle_target";
    const connect = () => {
        const next = document.querySelector(`${selector} .js-plotly-plot`);
        if (chart) chart.removeListener?.("plotly_relayout", onRelayout);
        chart = next && typeof next.on === "function" ? next : null;
        if (chart) chart.on("plotly_relayout", onRelayout);
    };
    const observer = new MutationObserver(connect);
    observer.observe(document.body, { childList: true, subtree: true });
    const prepareDraw = (event) => {
        if (event.target.closest?.(selector)) connect();
    };
    document.addEventListener("pointerdown", prepareDraw, true);
    connect();
    return () => {
        observer.disconnect();
        document.removeEventListener("pointerdown", prepareDraw, true);
        if (chart) chart.removeListener?.("plotly_relayout", onRelayout);
    };
}
