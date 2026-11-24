# Results summary

All numbers come from `python -m src.model`. The full output is in `reports/model_results.txt` and the walk-through is in `notebooks/rent_model.ipynb`.

**Data.** There are 12,166 Midwest listings with price, bedrooms, bathrooms and square footage. Of these, 468 are flagged and 730 are in the Chicago metro.

**Split.** 20% of rows are held out, stratified on the Chicago metro indicator.

**Features.**

- Predictors: log square footage, bedrooms, bathrooms, haversine distance from campus, amenity count, one indicator per common amenity (16 amenities appear in at least 5% of listings), pet policy, state, and a Chicago metro indicator.
- Target: log rent. Skew is 3.76 before the log transform and 0.56 after. Square footage is also logged (skew 5.28 before, 0.23 after). Distance is left unlogged, because logging made its skew worse.
- Price per square foot and cost per bedroom are engineered, but they are not used as predictors. Both are computed from rent, so using them would leak the answer into the inputs. Cost per bedroom is used in the match score instead.

## Model comparison

Held-out set, with models trained excluding flagged rows (2,331 rows):

| Model | RMSE | MAE | R² | 5-fold CV RMSE | 5-fold CV R² |
|---|---|---|---|---|---|
| Baseline (mean rent) | $537 | $379 | 0.000 | – | – |
| Multiple linear regression | $391 | $254 | 0.469 | $384 ± 26 | 0.508 ± 0.035 |
| Random forest | $301 | $173 | 0.685 | $308 ± 41 | 0.686 ± 0.035 |

The same models on held-out subsets:

| Subset | Rows | Baseline RMSE | Linear RMSE (R²) | Forest RMSE (R²) |
|---|---|---|---|---|
| Chicago metro | 130 | $969 | $703 (0.19) | $507 (0.58) |
| Outside Chicago metro | 2,201 | $500 | $365 (0.47) | $284 (0.68) |
| Within 25 miles of campus | 115 | $1,026 | $718 (0.15) | $527 (0.54) |

**Including vs excluding flagged rows.** On the same 2,331 unflagged held-out rows:

- Linear regression: RMSE $391 with or without flagged rows.
- Random forest: RMSE $301 without flagged rows, $300 with them.

The random forest wins in both variants, so the flagged rows do not change any conclusion. The deployed model is trained without them.

**Chicago metro fully held out.** Here the models are trained on non-Chicago rows only and tested on all 664 unflagged Chicago metro rows.

| Model | RMSE | R² |
|---|---|---|
| Midwest mean baseline | $1,130 | −0.39 |
| Linear regression | $979 | −0.04 |
| Random forest | $1,054 | −0.21 |

Without Chicago rows in training, neither model predicts Chicago rents better than a flat average. The model's Chicago accuracy depends on the roughly 530 Chicago listings it does see in training.

## Error and price

Dollar error grows with rent. For the random forest:

- Mean absolute error is $87 to $154 for actual rents under $1,410, and $399 above that.
- The rank correlation between absolute error and predicted rent is 0.36.

The residual plot in `reports/figures/residuals.png` shows the funnel shape. Percentage error is steadier: the median is 7% to 15% across rent bands. This is the reason for modelling log rent.

## Fair rent range

The deployed estimate is the random forest with an 80% prediction interval.

- **How the interval is built.** It comes from out-of-fold residuals on the training set, calibrated separately in five bands of predicted rent.
- **Coverage on held-out data.** Overall coverage is 79.7%. In each band of predicted rent it is between 78% and 82%.
- **Width.** The median width is $470, which is 48% of the estimate. The linear model's interval would be $682 wide (71%).
- **Where it misses.** By actual rent, coverage is lower at the extremes: 74% for the cheapest fifth and 67% for the most expensive fifth. Unusually cheap or expensive listings are partly what the model cannot explain.
- **How estimates are produced.** Every listing in the app gets an out-of-fold estimate, so no listing is judged by a model that was trained on its own rent.

Value labels across all 12,166 listings: 9,774 within, 1,208 below, 1,184 above.

## What drives rent

| Feature | Linear effect at median rent | Forest importance (held-out RMSE rise when shuffled) |
|---|---|---|
| Square footage | +$54 per 10% more space | $181 |
| Distance from campus | −$3 per 10 more miles | $121 |
| Bathrooms | +$195 per extra bathroom | $90 |
| Bedrooms (same square footage) | −$53 per extra bedroom | $40 |
| State | MN +$346, WI +$126, MI +$104, KS −$114 vs OH | $27 |
| Chicago metro | +$462 vs elsewhere in Illinois | $20 |
| Amenity count | +$80 per listed amenity | $10 |

**Size, bathrooms and location matter most.**

- Size dominates both models.
- Distance from campus ranks second in the random forest but near zero in the linear model. The forest uses distance as a stand-in for location: which metro area and which part of a state. A straight line in miles cannot capture that.
- Read distance as "where in the Midwest", not as "closer to campus is cheaper".

**Amenities.** Individual amenity effects are small and often negative. Playground is −$121 and Gym is +$93, each including its contribution to the amenity count. These reflect which kinds of complexes advertise which amenities, not what an amenity is worth. All effects are associations in 2018–2019 listings, not causes.

## Limitations

- **Data age.** The listings are from 2018 and 2019. Rents have risen substantially since then, so every dollar figure, including the fair rent range, is historical. The tool is suited to comparing listings with each other, not to budgeting for the current year.
- **Coordinate resolution.**
  - There are 7,364 distinct coordinate pairs across 98,043 rows nationally.
  - Neighbouring distinct coordinates are typically about 2 miles apart (median nearest-neighbour distance 2.15 miles).
  - Locations therefore mark areas, not buildings. Distance from campus is used only as a coarse banded filter and score, never as a fine ranking signal.
- **Midwest scope.**
  - The model is trained on 12 Midwest states, because the Chicago metro alone has only about 700 usable listings.
  - Most of the training data comes from markets far from campus, so the model learns general Midwest pricing first.
  - When the Chicago metro is held out completely, both models fail to beat a flat average. Local accuracy therefore rests on a few hundred Chicago-area rows.
- **Untested local accuracy near campus.**
  - Only 115 held-out listings fall within 25 miles of campus. On them the random forest gets RMSE $527 and R² 0.54, which is noticeably worse than its Midwest-wide R² of 0.69.
  - No listings from the immediate campus area have been checked against current or ground-truth rents.
  - Until they are, treat the fair rent range near campus as a rough guide.
- **Missing fields.** About 51% of Midwest listings have no pet policy and about 19% have no amenity list. The match score treats these as unknown, scored neutral with lower confidence. It does not treat them as negatives.
