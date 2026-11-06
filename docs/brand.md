---
title: Brand Assets
description: Download the official UniFace logo and icon in SVG and PNG formats.
---

# Brand Assets

Official UniFace logos and icons. Use them freely when you write about UniFace, link to it, or show that your project uses it.

[Download all assets (.zip)](assets/brand/uniface-brand-assets.zip){ .md-button .md-button--primary download }

---

## Logo

Use the **light** version on white or light backgrounds and the **dark** version on black or dark backgrounds.

<div class="grid" markdown>

<div markdown>

<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:40px 24px;text-align:center">
  <img src="../assets/brand/logo/uniface-logo-light.svg" alt="UniFace logo for light backgrounds" width="300">
</div>

**For light backgrounds**

[SVG](assets/brand/logo/uniface-logo-light.svg){ download } ·
[PNG 1200 px](assets/brand/logo/uniface-logo-light-1200.png){ download } ·
[PNG 2400 px](assets/brand/logo/uniface-logo-light-2400.png){ download }

</div>

<div markdown>

<div style="background:#0d1117;border:1px solid #0d1117;border-radius:12px;padding:40px 24px;text-align:center">
  <img src="../assets/brand/logo/uniface-logo-dark.svg" alt="UniFace logo for dark backgrounds" width="300">
</div>

**For dark backgrounds**

[SVG](assets/brand/logo/uniface-logo-dark.svg){ download } ·
[PNG 1200 px](assets/brand/logo/uniface-logo-dark-1200.png){ download } ·
[PNG 2400 px](assets/brand/logo/uniface-logo-dark-2400.png){ download }

</div>

</div>

---

## Icon

Use the icon alone where space is small: avatars, app icons, favicons.

<div style="background:#f6f8fa;border-radius:12px;padding:32px 24px;text-align:center">
  <img src="../assets/brand/icon/uniface-icon.svg" alt="UniFace icon" width="128">&nbsp;&nbsp;&nbsp;
  <img src="../assets/brand/icon/uniface-icon.svg" alt="" width="64">&nbsp;&nbsp;&nbsp;
  <img src="../assets/brand/icon/uniface-icon.svg" alt="" width="32">&nbsp;&nbsp;&nbsp;
  <img src="../assets/brand/icon/uniface-icon.svg" alt="" width="16">
</div>

| File | Size | Best for |
|---|---|---|
| [uniface-icon.svg](assets/brand/icon/uniface-icon.svg){ download } | Any size | Websites, documents |
| [uniface-icon-1024.png](assets/brand/icon/uniface-icon-1024.png){ download } | 1024 × 1024 | Large displays, print |
| [uniface-icon-512.png](assets/brand/icon/uniface-icon-512.png){ download } | 512 × 512 | Avatars (GitHub, PyPI, Hugging Face) |
| [uniface-icon-180.png](assets/brand/icon/uniface-icon-180.png){ download } | 180 × 180 | Phone home-screen icon |
| [favicon.ico](assets/brand/icon/favicon.ico){ download } | 16, 32, 48, 64 px | Browser tabs |

---

## Which file should I use?

| You need the logo for… | Use |
|---|---|
| A website or README | SVG (always sharp, smallest file) |
| A blog post or docs page that only accepts images | PNG 1200 px |
| Slides, posters, or print | PNG 2400 px |

---

## Colors

| Color | Hex | Used for |
|---|---|---|
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#EE5A2F;vertical-align:middle"></span> UniFace orange | `#EE5A2F` | Icon tile, upper half |
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#C9431F;vertical-align:middle"></span> Deep orange | `#C9431F` | Icon tile, lower half |
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#FFFFFF;border:1px solid #d0d7de;vertical-align:middle"></span> White | `#FFFFFF` | Icon plate, upper half |
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#FFD9CC;vertical-align:middle"></span> Peach | `#FFD9CC` | Icon plate, lower half |
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#0B0F17;vertical-align:middle"></span> Ink | `#0B0F17` | Name on light backgrounds |
| <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:#F0F3F6;border:1px solid #d0d7de;vertical-align:middle"></span> Off-white | `#F0F3F6` | Name on dark backgrounds |

---

## Usage guidelines

- **Keep space around the logo.** Leave at least half the icon's height empty on every side.
- **Keep it readable.** Don't show the full logo narrower than 120 px; use the icon alone instead.
- **Don't change it.** Don't stretch, rotate, recolor, add effects, or rearrange the icon and the name.

---

## Add the logo to your README

Show that your project uses UniFace. This snippet switches between the light and dark versions automatically on GitHub:

```html
<a href="https://github.com/yakhyo/uniface">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://yakhyo.github.io/uniface/assets/brand/logo/uniface-logo-dark.svg">
    <img src="https://yakhyo.github.io/uniface/assets/brand/logo/uniface-logo-light.svg" alt="UniFace" width="200">
  </picture>
</a>
```
