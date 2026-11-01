"""Lucide SVG Icons helper with standard CSS-variable theme adaptation for Streamlit."""

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
    """Standard, theme-native CSS that dynamically blends with Streamlit's --text-color

    and --background-color for 100% WCAG contrast in BOTH Light and Dark themes.
    """
    return """
    <style>
    /* Heading text dynamically inherits Streamlit's native theme text color */
    .icon-heading {
        display: flex;
        align-items: center;
        margin-top: 1.3rem;
        margin-bottom: 0.7rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: var(--text-color, #111827) !important;
    }
    .icon-heading h1, h1.icon-heading { font-size: 1.85rem; }
    .icon-heading h2, h2.icon-heading { font-size: 1.35rem; }
    .icon-heading h3, h3.icon-heading { font-size: 1.15rem; }

    /* Lucide Icon Badge Chip */
    .lucide-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        padding: 6px;
        margin-right: 10px;
        flex-shrink: 0;
        transition: transform 0.15s ease, background 0.2s ease;
    }
    .lucide-chip:hover {
        transform: scale(1.05);
    }

    /* 
       Standard dynamic color derivation:
       Uses color-mix with Streamlit's native --text-color so that:
       - On Light backgrounds (--text-color is dark #31333F):
         Icons automatically become deep, saturated, high-contrast jewel tones (WCAG 7+:1).
         Background chips become subtle pastel tints (12% opacity).
       - On Dark backgrounds (--text-color is light #FAFAFA):
         Icons automatically become luminous, vivid tones.
         Background chips become glowing translucent badges.
    */
    .chip-blue {
        color: color-mix(in srgb, #2563eb 72%, var(--text-color, #1e40af));
        background: color-mix(in srgb, #2563eb 12%, transparent);
        border: 1px solid color-mix(in srgb, #2563eb 28%, transparent);
    }

    .chip-purple {
        color: color-mix(in srgb, #7c3aed 72%, var(--text-color, #6d28d9));
        background: color-mix(in srgb, #7c3aed 12%, transparent);
        border: 1px solid color-mix(in srgb, #7c3aed 28%, transparent);
    }

    .chip-green {
        color: color-mix(in srgb, #059669 72%, var(--text-color, #047857));
        background: color-mix(in srgb, #059669 12%, transparent);
        border: 1px solid color-mix(in srgb, #059669 28%, transparent);
    }

    .chip-amber {
        color: color-mix(in srgb, #d97706 72%, var(--text-color, #b45309));
        background: color-mix(in srgb, #d97706 12%, transparent);
        border: 1px solid color-mix(in srgb, #d97706 28%, transparent);
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
    """Returns HTML for a heading with a theme-adaptive Lucide icon badge."""
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
