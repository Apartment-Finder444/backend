# Using Campus Rent Finder

## Before you start: these are old listings

Every listing is from 2018 or 2019. The rents are what landlords asked back then, so they are not today's prices. The tool is good for comparing listings with each other. It cannot tell you what a unit costs now.

## The match score

The match score says how well a listing fits what you asked for, from 0% to 100%.

You set preferences such as a budget, bedrooms, bathrooms, minimum size, distance from campus, amenities and a per-person budget. Each preference becomes a small score between 0 and 1:

| Preference | How it is scored |
|---|---|
| Rent | Full marks up to 85% of your budget, then a gentle decline to your budget. A sharp drop just above it, reaching zero at 10% over. |
| Bedrooms and bathrooms | Full marks for an exact match, half marks if one off, zero beyond that |
| Square footage | Full marks at or above your minimum, falling to zero at 70% of it |
| Distance from campus | Scored in bands: about 3, 7, 15, 30 and 60 miles. It is not exact, because listing locations mark areas about a mile and a half across rather than buildings. |
| Amenities | The share of the amenities you asked for that the listing confirms |
| Cost per bedroom | Rent divided by bedrooms, with a studio counted as one, scored against your per-person budget the same way as rent |

Only the preferences you fill in are counted. They are combined as a weighted average. The weights come from your ranking in **What matters most**: the top item counts most and the bottom item least, and the form shows each item's percentage. Drag the items to reorder them, or use the arrow buttons.

## Confidence

Next to every match score is a **confidence** figure. It is the share of the score's weight that rests on facts the listing actually states.

- 100% means every preference you set could be checked against this listing.
- A lower number means some of what you care about is not mentioned in the listing.

When a listing does not state something, that part of the score is set to a neutral 50%. It does not get zero. A listing that says nothing about amenities is never ranked below a listing that confirms it lacks what you want. The lower confidence is how the app shows the difference.

## The fair rent range

Each listing shows an **estimated fair rent range**, for example $998 to $1,639.

- A random forest model estimates the range from the listing's size, bedrooms, bathrooms, location, amenities and pet policy. It learned from about 9,000 other Midwest listings.
- The range is an **80% range**. On listings the model had never seen, the real asking rent fell inside the range about 8 times in 10.
- The range is wide on purpose. A single number would look more precise than the data allows.
- Each listing's estimate comes from a version of the model that never saw that listing's own rent.

The badge on the card compares the asking rent with this range:

- **Below expected range**: the asking rent is under the low end. It may be a good deal, or the listing may be missing something the model cannot see.
- **Within expected range**: the asking rent is in line with similar listings. Most listings land here.
- **Above expected range**: the asking rent is over the high end. Rent that is only above the middle of the range does not count.
- **No estimate available**: the listing lacks details the model needs. That says nothing about whether its rent is fair.

The estimates are weakest for expensive listings and for the Chicago area. See `reports/results_summary.md` for details.

## Why some fields show as unknown

Many listings leave things out:

- About half (51%) do not state a pet policy.
- About 19% have no amenity list.
- A few have no usable map location.

Unknown fields are shown in a dashed, grey, italic style, with a line such as "Not stated in this listing: pet policy". That is deliberately different from a confirmed no. "No pets allowed" appears as a solid red chip.

- **Pet policy not listed** means the landlord did not say. Ask before ruling the place in or out.
- **Amenities not listed** means there is no amenity list. The unit may still have them.
- **Amenity not mentioned** means the listing has an amenity list but your amenity is not on it.

If you have pets, listings with an unstated policy still appear, below the ones that confirm your pets are welcome. Tick **Only show listings that state they allow pets** to hide them. Listings that explicitly say no pets are always removed.

## Data quality flags

Some listings carry a short "Data check" note, for example a title that says studio on a two-bedroom listing, or a location far from the stated city. These listings are kept because the problem may be minor. Tick **Hide listings with data quality flags** to remove them.
