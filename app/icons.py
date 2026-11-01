"""Lucide SVG Icons helper with high-contrast, theme-adaptive styling for Streamlit."""

from typing import Optional


LUCIDE_SVG_PATHS = {
    "mic": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    "audio-lines": '<path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "refresh-cw": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "upload-cloud": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>',
    "headphones": '<path d="M3 14h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "volume-2": '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "radio": '<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/>',
    "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/>',
}


def get_theme_styles() -> str:
    """Returns CSS ensuring perfect visibility, contrast, and badge chips in both light & dark modes."""
    return """
    <style>
    /* CSS Variables for Light Mode */
    :root {
        --lucide-hdr-color: #0f172a;
        --lucide-sub-color: #475569;
        
        --badge-blue-fg: #1d4ed8;
        --badge-blue-bg: #eff6ff;
        --badge-blue-border: #bfdbfe;

        --badge-purple-fg: #6d28d9;
        --badge-purple-bg: #f5f3ff;
        --badge-purple-border: #ddd6fe;

        --badge-green-fg: #047857;
        --badge-green-bg: #ecfdf5;
        --badge-green-border: #a7f3d0;

        --badge-amber-fg: #b45309;
        --badge-amber-bg: #fffbeb;
        --badge-amber-border: #fde68a;
    }

    /* Dark Mode: triggered by OS prefers-color-scheme or Streamlit dark theme container */
    @media (prefers-color-scheme: dark) {
        :root {
            --lucide-hdr-color: #f8fafc;
            --lucide-sub-color: #94a3b8;

            --badge-blue-fg: #93c5fd;
            --badge-blue-bg: rgba(30, 58, 138, 0.4);
            --badge-blue-border: rgba(147, 197, 253, 0.35);

            --badge-purple-fg: #c4b5fd;
            --badge-purple-bg: rgba(76, 29, 149, 0.4);
            --badge-purple-border: rgba(196, 181, 253, 0.35);

            --badge-green-fg: #6ee7b7;
            --badge-green-bg: rgba(6, 78, 59, 0.4);
            --badge-green-border: rgba(110, 231, 183, 0.35);

            --badge-amber-fg: #fcd34d;
            --badge-amber-bg: rgba(120, 53, 15, 0.4);
            --badge-amber-border: rgba(252, 211, 77, 0.35);
        }
    }

    /* Fallback Streamlit explicit dark theme attribute */
    [data-theme="dark"],
    [data-testid="stAppViewContainer"].dark,
    .stApp[data-theme="dark"] {
        --lucide-hdr-color: #f8fafc !important;
        --lucide-sub-color: #94a3b8 !important;

        --badge-blue-fg: #93c5fd !important;
        --badge-blue-bg: rgba(30, 58, 138, 0.4) !important;
        --badge-blue-border: rgba(147, 197, 253, 0.35) !important;

        --badge-purple-fg: #c4b5fd !important;
        --badge-purple-bg: rgba(76, 29, 149, 0.4) !important;
        --badge-purple-border: rgba(196, 181, 253, 0.35) !important;

        --badge-green-fg: #6ee7b7 !important;
        --badge-green-bg: rgba(6, 78, 59, 0.4) !important;
        --badge-green-border: rgba(110, 231, 183, 0.35) !important;

        --badge-amber-fg: #fcd34d !important;
        --badge-amber-bg: rgba(120, 53, 15, 0.4) !important;
        --badge-amber-border: rgba(252, 211, 77, 0.35) !important;
    }

    /* Header styling */
    .icon-heading {
        display: flex;
        align-items: center;
        margin-top: 1.25rem;
        margin-bottom: 0.65rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: var(--lucide-hdr-color);
    }
    .icon-heading h1 { font-size: 1.85rem; }
    .icon-heading h2 { font-size: 1.35rem; }
    .icon-heading h3 { font-size: 1.15rem; }

    /* Badge chip wrapper */
    .lucide-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        padding: 6px;
        margin-right: 11px;
        flex-shrink: 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease;
    }
    .lucide-chip:hover {
        transform: scale(1.04);
    }

    .chip-blue {
        color: var(--badge-blue-fg);
        background: var(--badge-blue-bg);
        border: 1px solid var(--badge-blue-border);
    }
    .chip-purple {
        color: var(--badge-purple-fg);
        background: var(--badge-purple-bg);
        border: 1px solid var(--badge-purple-border);
    }
    .chip-green {
        color: var(--badge-green-fg);
        background: var(--badge-green-bg);
        border: 1px solid var(--badge-green-border);
    }
    .chip-amber {
        color: var(--badge-amber-fg);
        background: var(--badge-amber-bg);
        border: 1px solid var(--badge-amber-border);
    }
    </style>
    """


def lucide_svg(
    name: str,
    size: int = 20,
    color: str = "currentColor",
    stroke_width: float = 2.0,
    extra_style: str = "",
) -> str:
    """Returns standalone inline SVG element from Lucide icons."""
    paths = LUCIDE_SVG_PATHS.get(name, LUCIDE_SVG_PATHS["sparkles"])
    style_attr = f'style="vertical-align: middle; display: block; {extra_style}"'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" {style_attr}>{paths}</svg>'
    )


def icon_header(
    title: str,
    icon_name: str,
    level: int = 2,
    variant: str = "blue",
    size: int = 20,
) -> str:
    """Returns HTML for a heading with a crisp, theme-aware Lucide icon badge."""
    svg = lucide_svg(icon_name, size=size, color="currentColor")
    tag = f"h{level}"
    font_sizes = {1: "1.9rem", 2: "1.35rem", 3: "1.15rem"}
    f_size = font_sizes.get(level, "1.2rem")

    return (
        f'<{tag} class="icon-heading" style="font-size: {f_size};">'
        f'<span class="lucide-chip chip-{variant}">{svg}</span>'
        f'<span>{title}</span>'
        f'</{tag}>'
    )
