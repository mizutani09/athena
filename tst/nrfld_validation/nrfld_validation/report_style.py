"""Shared inline typography for portable validation HTML reports."""

REPORT_TYPOGRAPHY = """
:root {font-family:system-ui,sans-serif;font-size:21px;line-height:1.65}
body {font-size:1rem}
h1,h2,h3 {line-height:1.3;margin-top:0}
h1 {font-size:2.1rem} h2 {font-size:1.55rem} h3 {font-size:1.2rem}
table {font-size:.96rem}
figcaption {font-size:1rem;font-weight:600}
@media(max-width:700px) {:root {font-size:19px}}
"""
