"""All user-facing text in one place: page copy, glossary, class names, errors.

Short sentences, no jargon without a tooltip.
"""

from __future__ import annotations

APP_TITLE = "Planet Hunter"
TAGLINE = ("Find planets around other stars, and figure out what's really going on when it "
           "isn't a planet.")

STEPS = [
    ("Pick a star", "Choose one of our examples, or any star NASA's TESS telescope has watched."),
    ("We analyse its brightness", "We look for tiny dips that repeat, then check what could "
                                  "cause them."),
    ("You get a verdict", "A planet, two stars eclipsing each other, a star that changes on its "
                          "own, or just noise."),
]

EXPLAINER_TITLE = "The idea in 30 seconds"
EXPLAINER = ("Stars look steady, but a space telescope can measure their brightness very "
             "precisely. When a planet passes in front of its star, it blocks a little light, and "
             "the star dims for a few hours. If that dip repeats like clockwork, something may be "
             "orbiting the star. But other things cause dips too, so we check the shape, depth and "
             "timing of every dip before calling it a planet.")
EXPLAINER_CAPTION = "A planet passing in front of its star blocks a little light."

# Class name -> (label, colour, text colour on that colour, one-line meaning)
# Colours: a colourblind-safe palette that reads on light and dark backgrounds;
# every colour is always shown together with its text label.
CLASS_INFO = {
    "PLANET": ("Planet", "#0072B2", "#FFFFFF",
               "A planet passing in front of its star."),
    "EB": ("Eclipsing binary", "#D55E00", "#FFFFFF",
           "Two stars orbiting each other, taking turns to block each other's light."),
    "VARIABLE": ("Variable star", "#007A5E", "#FFFFFF",
                 "A star whose brightness changes on its own: pulsing, or spots turning."),
    "NOISE": ("Noise", "#6B6B6B", "#FFFFFF",
              "Nothing real: random flicker or effects of the telescope."),
    "NEEDS_REVIEW": ("Needs review", "#CC79A7", "#1A1A1A",
                     "We're not sure: kept for a closer look rather than thrown away."),
}
OUTCOME_ORDER = ["PLANET", "EB", "VARIABLE", "NOISE"]
SEE_EXAMPLE = "See an example"
TRY_EXAMPLE = "Try an example"

GLOSSARY = {
    "light curve": "A graph of a star's brightness over time.",
    "transit": "A planet passing in front of its star, which dims the star slightly.",
    "eclipse": "One star passing in front of another and blocking some of its light.",
    "eclipsing binary": "Two stars orbiting each other, lined up so that they take turns "
                        "blocking each other's light.",
    "variable star": "A star whose brightness changes on its own, for example because it "
                     "pulsates or has dark spots.",
    "period": "The time between two repeats of the same thing, such as two dips.",
    "sector": "TESS watches one patch of sky for about 27 days; each of these stretches is "
              "called a sector.",
    "TIC ID": "The star's number in the TESS Input Catalog, a list of about a billion stars.",
    "confidence": "How sure the model is: High (80% or more), Medium (50-80%), Low (under 50%).",
    "false positive": "A signal that looks like a planet at first but turns out to be "
                      "something else.",
    "TOI": "TESS Object of Interest: a star NASA flagged because its light shows possible "
           "planet dips.",
}

# ── Analyze page ─────────────────────────────────────────────
STEP_NAMES = ["Choose star", "Settings", "Results"]
PICK_HOW = "How do you want to choose a star?"
PICK_EXAMPLE = "Pick an example"
PICK_TOI = "Search all TESS candidates"
PICK_MANUAL = "I know the TIC ID / star name"
EXAMPLE_LABEL = "Example stars"
EXAMPLE_HELP = "Hand-picked stars with clear data. Each one loads instantly."
TOI_LABEL = "TESS candidates (TOIs)"
TOI_HELP = "Every star NASA has flagged as a possible planet host. Type to search."
MANUAL_LABEL = "Star name or TIC ID"
MANUAL_HELP = "For example 'HD 209458', 'TIC 420814525' or just '420814525'."
MANUAL_FIND = "Find this star"
RANDOM_EXAMPLE = "Random example"
FOUND = "Found: {what}"
DEMO_BADGE = "Precomputed example: click Re-run to analyse it live."
RERUN = "Re-run live"
ANALYZE = "Analyze"
ANALYZE_ANOTHER = "Analyze another star"

PRESETS = {
    "quick": ("Quick", "Up to 3 sectors, fastest."),
    "standard": ("Standard", "Up to 10 sectors (recommended)."),
    "thorough": ("Thorough", "All sectors, plus a second search method and up to 3 signals. "
                             "Slowest."),
}
PRESET_LABEL = "How thorough?"
ESTIMATE = "About {minutes} to analyse, downloading about {mb:.0f} MB."
ADVANCED = "Advanced settings"
ADV_PERIOD = "Search periods (days)"
ADV_PERIOD_HELP = "Shortest and longest time between dips to look for."
ADV_SECTORS = "Sectors to use"
ADV_SECTORS_HELP = "Leave empty to let the preset choose."
ADV_WINDOW = "Smoothing window (days)"
ADV_WINDOW_HELP = ("How much slow brightness change to remove before searching for dips. "
                   "Longer keeps more of long dips.")
ADV_TLS = "Also use the second search method (slower)"
ADV_TLS_HELP = "A search that uses a realistic dip shape. Off by default."
ADV_SIGNALS = "Signals to look for"
ADV_SIGNALS_HELP = "After finding the strongest repeating dip, look for weaker ones too."

STAGES = {
    "Looking up the star": "Looking up the star in the catalog...",
    "Cleaning the data": "Cleaning the data...",
    "Searching for repeating dips": "Searching for repeating dips...",
    "Re-cleaning around the dips": "Re-checking the data around the dips...",
    "Checking whether it's really a planet": "Checking whether it's really a planet...",
    "Classifying the signal": "Asking the classifier...",
    "Working out what it is": "Working out what it is...",
    "Writing your report": "Writing your report...",
}
STAGE_DOWNLOAD = "Downloading TESS data ({detail})..."
DONE = "Analysis complete."

STAR_CARD_TITLE = "About this star"
STAR_BRIGHTNESS = "Brightness (TESS magnitude)"
STAR_BRIGHTNESS_HELP = "Smaller numbers are brighter. Stars below about 12 give clean data."
STAR_TEMP = "Temperature"
STAR_SIZE = "Size"
STAR_SECTORS = "TESS sectors available"
STAR_SECTORS_HELP = GLOSSARY["sector"]

VERDICT_TITLE = "Verdict"
WHY_TITLE = "Why do we think so?"
CATALOG_TITLE = "What the catalogs say"
FACTS_TITLE = "Key facts"
CHARTS_TITLE = "Charts"
WHAT_AM_I = "What am I looking at?"
MODEL_LEANED = ("On the light-curve measurements alone the classifier leaned towards {model}; "
                "the physics check above overrides that.")
RAW_PROBS = "Classifier probabilities"

CATALOG_AGREE = "{source} lists this as {what}: we agree."
CATALOG_DISAGREE = "{source} lists this as {what}: we disagree. See the Tricky cases group."
CATALOG_UNSURE = "{source} lists this as {what}. We're not sure yet, so it's kept for review."
CATALOG_NONE = "This star is not in the catalogs we checked."

CHARTS = {
    "lightcurve": (
        "The star's brightness over time",
        "Each dot is one brightness measurement. Shaded bands are separate TESS sectors "
        "(about 27 days each). Red ticks mark where the repeating dips we found should fall. "
        "Slow wiggles are the star itself; the dips we care about are short and repeat."),
    "fold": (
        "All dips stacked on top of each other",
        "We cut the light curve into pieces, one per orbit, and stack them so every dip lines "
        "up at the centre. Grey dots are measurements, the dark line is their average, and the "
        "coloured line is the dip shape we fitted. Stacking makes a dip far clearer than any "
        "single one."),
    "planet": (
        "How big is it?",
        "The size we estimate, next to Earth, Neptune and Jupiter. The error bar shows how "
        "uncertain it is, mostly because we don't know the star's size exactly."),
    "eb": (
        "The two eclipses side by side",
        "Left: the deeper eclipse, when the brighter star is hidden. Right: the same stretch "
        "half an orbit later, when the fainter star is hidden. Two eclipses of different depth "
        "are the signature of two stars; a planet gives almost no second dip."),
    "variable": (
        "Which rhythms are in the light?",
        "Each peak is a regular rhythm in the star's brightness; higher peaks are stronger. "
        "The marked peak is the star's main period."),
}

EXPERT_TABS = ["Detection", "Vetting", "Variability", "Model", "Data"]
DOWNLOADS = "Download"
DL_HTML = "Report (HTML)"
DL_JSON = "Data (JSON)"
DL_CSV = "Summary (CSV)"
SHARE = "Share link"
SHARE_HELP = "Add this to the app's address to open this star directly:"
RECENT = "Recent analyses"
RECENT_EMPTY = "Stars you analyse appear here."
MODE_TOGGLE = "Expert mode"
MODE_HELP = "Show the technical charts, numbers and probabilities."

# ── Errors (title, body, suggestion) ─────────────────────────
ERRORS = {
    "unknown_star": ("We couldn't find that star",
                     "No star with that name or number is in the TESS catalog.",
                     "Check the spelling, try its TIC number, or pick one of our examples."),
    "no_data": ("No TESS data for this star",
                "TESS has not measured this star's brightness (or not with the data types we "
                "use).",
                "Try one of our examples, or another star from the TESS candidate list."),
    "timeout": ("NASA's archive didn't answer in time",
                "The download from the archive timed out. This is usually temporary.",
                "Wait a minute and try again, or open a precomputed example meanwhile."),
    "model_missing": ("The classifier isn't set up yet",
                      "The analysis works, but there is no trained model to give a verdict.",
                      "Build it with:\n\npython scripts/build_labels.py\n"
                      "python scripts/build_features.py\npython scripts/train_triage.py\n\n"
                      "The precomputed examples keep working meanwhile."),
    "generic": ("Something went wrong",
                "The analysis stopped unexpectedly.",
                "Try again, or pick one of our examples."),
}
TECH_DETAILS = "Show technical details"

# ── Plain-English reasons from the classifier's top features ─
# feature -> (low/"no" sentence, high/"yes" sentence, threshold); the value decides
# which one applies. Only reasons that support the final verdict are shown.
FEATURE_SENTENCES = {
    "odd_even_depth_ratio": ("Alternating dips have different depths.",
                             "Alternating dips have the same depth.", 0.9),
    "secondary_to_primary_ratio": ("There is no second, smaller dip halfway between, which "
                                   "eclipsing stars usually show.",
                                   "A second, smaller dip shows up halfway between the main "
                                   "dips.", 0.05),
    "secondary_best_to_primary_ratio": ("No second dip appears anywhere between the main dips.",
                                        "A second dip appears between the main dips.", 0.05),
    "ingress_to_total_ratio": ("The dips are U-shaped like a planet's, not V-shaped like a "
                               "star's.",
                               "The dips are V-shaped, more like two stars grazing each other.",
                               0.25),
    "sector_depth_cv": ("The dip depth stays the same across all observing sectors.",
                        "The dip depth changes from sector to sector.", 0.2),
    "red_noise_beta": ("The light curve is clean on the timescale of the dips.",
                       "The light curve has a lot of slow, irregular wobble.", 1.5),
    "ls_power_1": ("There is no strong, smooth brightness cycle.",
                   "The star's brightness rises and falls in a smooth, regular cycle.", 0.3),
    "acf_peak_height": ("The brightness pattern does not repeat cleanly.",
                        "The brightness pattern repeats very regularly.", 0.3),
    "ls_period_matches_bls": ("The dips are unrelated to any smooth brightness cycle.",
                              "The smooth brightness cycle has the same rhythm as the dips, "
                              "typical of close pairs of stars.", 0.5),
    "depth_snr_per_transit": ("Each dip is faint, close to the noise.",
                              "Each dip is clearly visible on its own.", 5.0),
    "single_event_excess": ("All the dips look alike.",
                            "One dip is much stronger than the others.", 2.0),
    "amp_p2p_ppm": ("The star's own brightness changes are small.",
                    "The star's brightness changes a lot on its own.", 10_000),
    "skew": ("The star spends most of its time at normal brightness, with short drops.",
             "The star spends most of its time faint, with short brightenings.", 0.0),
    "trapezoid_chi2_vs_box_chi2": ("A dip with sloped edges fits much better than a sharp box.",
                                   "A sharp box fits about as well as a sloped dip.", 0.9),
}
DURATION_SENTENCE_OK = ("The dips last about as long as expected for something orbiting this "
                        "star.")
DURATION_SENTENCE_OFF = ("The dips last much {word} than expected for something orbiting "
                         "this star.")
SIZE_SENTENCE = "The dip implies an object about {size}."
GENERIC_SENTENCE = "The measurement \"{name}\" points towards {cls}."

FEATURE_NAMES = {
    "teff": "star temperature", "rad": "star size", "logg": "star surface gravity",
    "tmag": "star brightness", "contratio": "light from nearby stars",
    "period_d": "time between dips", "duration_hr": "dip duration", "depth_ppm": "dip depth",
    "kurtosis": "how peaked the brightness distribution is",
    "fourier_R21": "shape of the brightness cycle", "fourier_R31": "shape of the brightness cycle",
    "refl_amp_ppm": "brightening once per orbit", "ellip_amp_ppm": "brightening twice per orbit",
    "crowdsap_min": "share of light from the target star",
    "var_period_d": "period of the star's own variation",
    "ls_period_1": "main rhythm in the brightness",
}

FOOTER = "Data: NASA TESS mission via the MAST archive. Results are automated; see How It Works."
