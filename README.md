# Lensed AGN Forecasts

This repository uses [`slsim`](https://github.com/LSST-strong-lensing/slsim) to simulate and forecast the yields and physical properties of strongly lensed AGNs (quasars) across upcoming astronomical surveys, including Rubin LSST and the Roman Space Telescope (Medium, Wide, and NANCY tiers).

**Contacts:** Paras Sharma ([github@timedilatesme](https://github.com/timedilatesme), paras.sharma@stonybrook.edu)

## Stats of different samples

| Survey | Sky Area (deg²) | Total Lensed-AGN | Doubles | Triples | Quads | Quints | Quad Fraction | Image Sep Cuts (") | Mag Cuts (2nd Brightest Image) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LSST | 20,000 | 2,659 | 2,337 | 51 | 269 | 2 | 10.1% | 0.5 - 4.0 | i &lt; 23.3 |
| Roman Medium Tier | 2,415 | 3,228 | 2,876 | 84 | 262 | 6 | 8.1% | 0.2 - 4.0 | F106 &lt; 26.5, F129 &lt; 26.4, F158 &lt; 26.4 |
| Roman Medium Tier (2029) | 500 | 683 | 605 | 15 | 63 | 0 | 9.2% | 0.2 - 4.0 | F106 &lt; 26.5, F129 &lt; 26.4, F158 &lt; 26.4 |
| Roman Wide Tier | 2,702 | 3,579 | 3,177 | 89 | 311 | 2 | 8.7% | 0.2 - 4.0 | F158 &lt; 26.2 |
| Roman NANCY | 1,000 | 1,011 | 882 | 26 | 103 | 0 | 10.2% | 0.2 - 4.0 | F158 &lt; 25.5 |
| Euclid Wide | 14,500 | 16,718 | 14,633 | 390 | 1,673 | 22 | 10.0% | 0.2 - 4.0 | VIS &lt; 26.2 |
| Euclid Wide (VIS+NISP) | 14,500 | 7,580 | 6,423 | 192 | 950 | 15 | 12.5% | 0.2 - 4.0 | VIS &lt; 26.2, Y &lt; 24.5, J &lt; 24.5, H &lt; 24.5 |

## Images of different samples

![LSST Montage](figures/montage_LSST.png)

![Roman Medium Tier Montage](figures/montage_Roman%20Medium%20Tier.png)

![Roman Wide Tier Montage](figures/montage_Roman%20Wide%20Tier.png)

![Roman NANCY Montage](figures/montage_Roman%20NANCY.png)

![Euclid Wide Montage](figures/montage_Euclid%20Wide.png)

## Plots of different samples

![LSST Roman Comparison](figures/corner_lsst_vs_roman_med_wide_combined.png)

![Roman Medium vs Wide](figures/corner_roman_med_vs_wide.png)

![Roman Medium vs NANCY](figures/corner_roman_nancy_vs_medium2029.png)

![LSST vs Roman Wide vs Euclid Wide](figures/corner_lsst_vs_roman_wide_vs_euclid_wide.png)

![Euclid Wide](figures/corner_euclid_wide.png)
