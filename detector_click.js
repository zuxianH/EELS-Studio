export default function ({ data, setTriggerValue }) {
    let chart = null;
    const onClick = (event) => {
        const point = event.points?.[0];
        if (!point || point.curveNumber !== 0) return;
        if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
        setTriggerValue("clicked", {
            px: point.y,
            py: point.x,
            preview_id: data.preview_id,
            event_id: crypto.randomUUID(),
        });
    };
    const connect = () => {
        const next = document.querySelector(".st-key-detector_click_target .js-plotly-plot");
        if (chart) chart.removeListener?.("plotly_click", onClick);
        chart = next && typeof next.on === "function" ? next : null;
        // Streamlit clears Plotly listeners during updates even if it reuses
        // the same DOM node. Reattach our handler after those DOM changes.
        if (chart) chart.on("plotly_click", onClick);
    };
    // Streamlit can replace the plot node on reruns or when a tab is mounted.
    const observer = new MutationObserver(connect);
    observer.observe(document.body, { childList: true, subtree: true });
    // React can clear listeners after the last DOM mutation. Restore the
    // handler in capture phase, before Plotly receives the next interaction.
    const prepareClick = (event) => {
        if (event.target.closest?.(".st-key-detector_click_target")) connect();
    };
    document.addEventListener("pointerdown", prepareClick, true);
    connect();
    return () => {
        observer.disconnect();
        document.removeEventListener("pointerdown", prepareClick, true);
        if (chart) chart.removeListener?.("plotly_click", onClick);
    };
}
