// Match the Detector/Spectrum card panels' heights by measuring the taller one and
// applying it as a min-height to both, since Streamlit's nested wrapper divs make a
// pure-CSS flex/percentage-height fix unreliable across versions.
export default function () {
    const selectors = [".st-key-detector_panel", ".st-key-spectrum_panel"];
    let frame = null;
    const equalize = () => {
        frame = null;
        const panels = selectors.map((s) => document.querySelector(s)).filter(Boolean);
        if (panels.length < 2) return;
        panels.forEach((panel) => { panel.style.minHeight = "0px"; });
        const tallest = Math.max(...panels.map((panel) => panel.getBoundingClientRect().height));
        panels.forEach((panel) => { panel.style.minHeight = `${tallest}px`; });
    };
    const schedule = () => {
        if (frame === null) frame = requestAnimationFrame(equalize);
    };
    // Only structural changes (not the min-height writes above, which touch no
    // children) can retrigger this, so it settles instead of looping.
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("resize", schedule);
    schedule();
    return () => {
        observer.disconnect();
        window.removeEventListener("resize", schedule);
        if (frame !== null) cancelAnimationFrame(frame);
    };
}
