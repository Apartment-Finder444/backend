# Campus Apartment Finder: results summary

All numbers come from `python -m src.model`.

- Full output: `reports/model_results.txt`
- Model walk-through: `notebooks/rent_model.ipynb`
- Exploratory analysis: `notebooks/eda.ipynb`

**Data.**

- There are 11,977 Midwest listings with price, bedrooms, bathrooms and square footage. Of these, 466 are flagged and 721 are in the Chicago metro.
- Cleaning followed the project proposal:
  - Weekly rents were converted to monthly.
  - Rents outside $200 to $20,000 were removed.
  - Exact duplicates and reposts of the same unit were removed.
  - The full record is in `docs/data_dictionary.md`.

**Split.** 20% of rows are held out, stratified on the Chicago metro indicator. Model selection uses 5-fold cross validation on the training set.

**Features.**

- Predictors: log square footage, bedrooms, bathrooms, haversine distance from campus, amenity count, one indicator per common amenity, cats and dogs indicators, state, and a Chicago metro indicator.
- Target: log rent. Skew is 3.76 before the log transform and 0.55 after. Square footage is also logged (skew 5.30 before, 0.23 after).
- Price per square foot and cost per bedroom are engineered, as the proposal describes. They are used for comparison in the app and in the match score, but not as predictors, because both are computed from rent.

## Model comparison

| Model | 5-fold CV RMSE | 5-fold CV MAE | 5-fold CV R² | Held-out RMSE | Held-out MAE | Held-out R² |
|---|---|---|---|---|---|---|
| Baseline (mean rent) | – | – | – | $620 | $386 | 0.00 |
| Multiple linear regression | $380 ± 19 | $250 ± 9 | 0.483 ± 0.024 | $419 | $255 | 0.542 |
| Random forest | $294 ± 21 | $173 ± 4 | 0.690 ± 0.025 | $388 | $174 | 0.608 |

**One listing dominates the held-out RMSE.**

- The held-out set (2,300 rows) contains a 4-bedroom, 6,000 sq ft Chicago listing at $15,000 a month. Its error alone is about $11,600.
- Without that one listing, the random forest's held-out RMSE is about $293, in line with cross validation. MAE barely moves either way.
- It stays in the data because it is inside the proposal's $200 to $20,000 range.
- This is the reason cross validation and MAE are the better guides to typical error, and RMSE is reported alongside them rather than alone.

**Random forest vs linear regression.**

- The random forest has lower error on every measure except held-out R² in Chicago.
- On the 130 held-out Chicago rows, linear regression gets R² 0.55 and the forest 0.36. The linear model extrapolates further for the $15,000 unit.
- Measured by MAE, the forest still wins in Chicago: $413 against $588.

| Held-out subset | Rows | Baseline RMSE | Linear RMSE (R²) | Forest RMSE (R²) | Forest MAE |
|---|---|---|---|---|---|
| Outside Chicago metro | 2,170 | $496 | $359 (0.47) | $281 (0.68) | $160 |
| Chicago metro | 130 | $1,640 | $976 (0.55) | $1,160 (0.36) | $413 |
| Within 25 miles of campus | 115 | $1,756 | $1,039 (0.52) | $1,238 (0.32) | $457 |

**Including vs excluding flagged rows.** On the same 2,300 unflagged held-out rows:

- Linear regression: RMSE $419 without flagged rows, $422 with them.
- Random forest: RMSE $388 without flagged rows, $385 with them.

The random forest wins on cross validation either way, so the conclusions do not change. The deployed model is trained without flagged rows.

**Chicago metro fully held out.** Here the models are trained on non-Chicago rows only and tested on all 655 unflagged Chicago rows.

| Model | RMSE | R² |
|---|---|---|
| Midwest mean baseline | $1,136 | −0.39 |
| Linear regression | $1,002 | −0.08 |
| Random forest | $1,075 | −0.25 |

Neither model beats a flat average there. The Chicago estimates depend on the roughly 520 Chicago listings the model sees in training.

## Error and price

The residuals show clear heteroscedasticity (`reports/figures/residuals.png`). For the random forest, mean absolute error by actual rent band is:

| Actual rent band | MAE |
|---|---|
| $325 to $735 | $121 |
| $735 to $1,415 (three middle bands) | $94 to $141 |
| $1,415 and above | $412 |

Percentage error is steadier, at 7% to 15% median. This is why the model works on log rent.

## Fair rent range

The deployed estimate is the random forest with an 80% prediction interval.

- **How the interval is built.** It comes from out-of-fold residuals, calibrated separately in five bands of predicted rent.
- **Coverage on held-out data.** Overall coverage is 81.2%. In every band of predicted rent it is between 80.9% and 82.0%.
- **Width.** The median width is $511, which is 48% of the estimate.
- **Where it misses.** Coverage is lower for listings that turn out to be unusually cheap or expensive: 72% in the bottom and top fifths of actual rent.
- **How estimates are produced.** Every listing in the app gets an out-of-fold estimate, so no listing is judged by a model that was trained on its own rent.
- **Checking a unit you describe.** The Flask app loads the serialized model (`models/random_forest.joblib`) to price new units through `/api/estimate` and the "Is this rent fair?" panel.

Value labels across all 11,977 listings: 9,663 within, 1,165 below, 1,149 above.

## What drives rent

| Feature | Linear effect at median rent | Forest importance (held-out RMSE rise when shuffled) |
|---|---|---|
| Square footage | +$54 per 10% more space | $154 |
| Distance from campus | −$3 per 10 more miles | $111 |
| Bathrooms | +$193 per extra bathroom | $87 |
| Bedrooms (same square footage) | −$56 per extra bedroom | $36 |
| State | MN highest, KS lowest, relative to OH | $24 |
| Chicago metro | +$503 vs elsewhere in Illinois | $24 |
| Amenity count | +$76 per listed amenity | $7 |

**Size, bathrooms and location matter most.**

- Distance ranks high in the random forest but near zero in the linear model. The forest uses it as a stand-in for which metro area and part of a state a listing is in, and a straight line in miles cannot capture that.
- Read distance as "where in the Midwest", not as "closer to campus is cheaper".

**Amenities.** Individual amenity effects are small and mixed: Parking +$98, Gym +$91, AC −$85, Playground −$119. They reflect which kinds of complexes advertise which amenities, not what an amenity is worth.

**Collinearity.** VIF is below 2.3 for every numeric predictor (`notebooks/eda.ipynb`).

## Limitations

- **Data age.** The listings are from 2018 and 2019, and rents have risen substantially since. Every dollar figure, including the fair rent range, is historical. The tool is suited to comparing listings with each other, not to budgeting for the current year.
- **Coordinate resolution.**
  - There are 7,363 distinct coordinate pairs across 96,714 rows nationally.
  - Neighbouring distinct coordinates are typically about 2 miles apart (median 2.15 miles).
  - Locations therefore mark areas, not buildings. Distance from campus is used only as a coarse, banded filter and score.
- **Midwest scope.**
  - The proposal planned a single metro area, but the Chicago metro has only about 700 usable listings. The model is therefore trained on 12 Midwest states, with state and a Chicago indicator.
  - Most of the training data comes from markets far from campus. Ohio alone is 39% of it.
  - With the Chicago metro held out completely, neither model beats a flat average.
- **Untested local accuracy near campus.**
  - Only 115 held-out listings fall within 25 miles of campus. On them the random forest gets MAE $457 and R² 0.32, well below its Midwest-wide performance.
  - No listings from the immediate campus area have been checked against current or ground-truth rents.
  - Until they are, treat the fair rent range near campus as a rough guide.
- **Missing fields.** 50.6% of Midwest listings have no pet policy and 18.9% have no amenity list. The match score treats these as unknown, scored neutral with lower confidence, never as negatives.
- **Extreme listings.** A handful of luxury units, such as the $15,000 Chicago listing, sit far outside the model's range and drive RMSE. Their fair rent ranges are the least reliable in the dataset.
