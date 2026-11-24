(function () {
  "use strict";

  const UNKNOWN_FIELD_LABELS = {
    pet_policy: "pet policy",
    amenities: "amenities",
    bedrooms: "bedroom count",
    bathrooms: "bathroom count",
    square_feet: "square footage",
    location: "map location",
    fair_rent: "fair rent estimate",
  };
  const VALUE_LABELS = {
    below: { text: "Below expected range", icon: "↓", className: "value-below" },
    within: { text: "Within expected range", icon: "↔", className: "value-within" },
    above: { text: "Above expected range", icon: "↑", className: "value-above" },
    unknown: { text: "No estimate available", icon: "?", className: "value-unknown" },
  };
  const LOW_CONFIDENCE = 0.75;
  const DRIVER_COUNT = 10;
  const EFFECT_COUNT = 10;

  const form = document.getElementById("search-form");
  const priorityList = document.getElementById("priority-list");
  const amenityOptions = document.getElementById("amenity-options");
  const stateSelect = document.getElementById("state-select");
  const resultsList = document.getElementById("results-list");
  const resultsSummary = document.getElementById("results-summary");

  let priorities = [];
  let draggedItem = null;

  function el(tag, attributes, children) {
    const node = document.createElement(tag);
    Object.entries(attributes || {}).forEach(([key, value]) => {
      if (value === null || value === undefined || value === false) {
        return;
      }
      if (key === "className") {
        node.className = value;
      } else if (key === "text") {
        node.textContent = value;
      } else if (key === "style") {
        Object.assign(node.style, value);
      } else {
        node.setAttribute(key, value === true ? "" : value);
      }
    });
    (children || []).forEach((child) => {
      if (child === null || child === undefined || child === false) {
        return;
      }
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  function money(value) {
    if (value === null || value === undefined) {
      return "n/a";
    }
    return "$" + Math.round(value).toLocaleString("en-US");
  }

  function signedMoney(value) {
    const rounded = Math.round(value);
    return (rounded >= 0 ? "+" : "−") + "$" + Math.abs(rounded).toLocaleString("en-US");
  }

  function percent(value) {
    return Math.round(value * 100) + "%";
  }

  function rankWeights(count) {
    const raw = Array.from({ length: count }, (_, index) => count - index);
    const total = raw.reduce((sum, value) => sum + value, 0);
    return raw.map((value) => value / total);
  }

  function renderPriorities() {
    priorityList.replaceChildren();
    const weights = rankWeights(priorities.length);
    priorities.forEach((priority, index) => {
      const up = el("button", { type: "button", "aria-label": "Move " + priority.label + " up", text: "▲", disabled: index === 0 });
      const down = el("button", { type: "button", "aria-label": "Move " + priority.label + " down", text: "▼", disabled: index === priorities.length - 1 });
      up.addEventListener("click", () => movePriority(index, index - 1));
      down.addEventListener("click", () => movePriority(index, index + 1));
      const item = el("li", { draggable: "true", "data-key": priority.key }, [
        el("span", { className: "priority-label", text: priority.label }),
        el("span", { className: "priority-weight", text: percent(weights[index]) }),
        up,
        down,
      ]);
      attachDrag(item);
      priorityList.appendChild(item);
    });
  }

  function movePriority(from, to) {
    if (to < 0 || to >= priorities.length) {
      return;
    }
    const [moved] = priorities.splice(from, 1);
    priorities.splice(to, 0, moved);
    renderPriorities();
    const button = priorityList.children[to].querySelectorAll("button")[from > to ? 0 : 1];
    if (button && !button.disabled) {
      button.focus();
    }
  }

  function attachDrag(item) {
    item.addEventListener("dragstart", (event) => {
      draggedItem = item;
      item.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", item.dataset.key);
    });
    item.addEventListener("dragend", () => {
      item.classList.remove("dragging");
      priorityList.querySelectorAll(".drop-target").forEach((node) => node.classList.remove("drop-target"));
      draggedItem = null;
    });
    item.addEventListener("dragover", (event) => {
      if (!draggedItem || draggedItem === item) {
        return;
      }
      event.preventDefault();
      item.classList.add("drop-target");
    });
    item.addEventListener("dragleave", () => item.classList.remove("drop-target"));
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      item.classList.remove("drop-target");
      if (!draggedItem || draggedItem === item) {
        return;
      }
      const from = priorities.findIndex((priority) => priority.key === draggedItem.dataset.key);
      const rect = item.getBoundingClientRect();
      const after = event.clientY > rect.top + rect.height / 2;
      const [moved] = priorities.splice(from, 1);
      let target = priorities.findIndex((priority) => priority.key === item.dataset.key);
      if (after) {
        target += 1;
      }
      priorities.splice(target, 0, moved);
      renderPriorities();
    });
  }

  function renderAmenities(amenities) {
    amenityOptions.replaceChildren(
      ...amenities.map((amenity) =>
        el("label", { className: "check" }, [
          el("input", { type: "checkbox", name: "amenity", value: amenity }),
          amenity,
        ])
      )
    );
  }

  function renderStates(states) {
    states.forEach((state) => stateSelect.appendChild(el("option", { value: state, text: state })));
  }

  function buildQuery() {
    const data = new FormData(form);
    const params = new URLSearchParams();
    const skip = new Set(["amenity", "confirmed_only"]);
    for (const [key, value] of data.entries()) {
      if (skip.has(key) || String(value).trim() === "") {
        continue;
      }
      params.set(key, String(value).trim());
    }
    const amenities = data.getAll("amenity");
    if (amenities.length) {
      params.set("amenities", amenities.join(","));
    }
    if (data.get("pets") && data.get("confirmed_only")) {
      params.set("confirmed_only", "true");
    }
    params.set("priorities", priorities.map((priority) => priority.key).join(","));
    return params;
  }

  function rangeGraphic(listing) {
    const low = listing.fair_rent_low;
    const high = listing.fair_rent_high;
    const price = listing.price;
    const minimum = Math.min(low, price) * 0.9;
    const maximum = Math.max(high, price) * 1.1;
    const span = maximum - minimum || 1;
    const position = (value) => ((value - minimum) / span) * 100 + "%";
    return el("div", {}, [
      el("div", { className: "range-graphic", role: "img", "aria-label": "Asking rent " + money(price) + " against estimated range " + money(low) + " to " + money(high) }, [
        el("div", { className: "range-track" }),
        el("div", { className: "range-band", style: { left: position(low), width: ((high - low) / span) * 100 + "%" } }),
        el("div", { className: "range-center", style: { left: position(listing.fair_rent) } }),
        el("div", { className: "range-marker", title: "Asking rent " + money(price), style: { left: position(price) } }),
      ]),
      el("div", { className: "range-scale" }, [
        el("span", { text: "Range " + money(low) + " to " + money(high) }),
        el("span", { text: "● asking " + money(price) }),
      ]),
    ]);
  }

  function fairRentSection(listing) {
    const label = VALUE_LABELS[listing.value_label] || VALUE_LABELS.unknown;
    const badge = el("span", { className: "badge " + label.className }, [el("span", { "aria-hidden": "true", text: label.icon }), label.text]);
    if (listing.value_label === "unknown" || listing.fair_rent_low === null) {
      return el("div", {}, [
        el("div", { className: "section-label", text: "Estimated fair rent" }),
        badge,
        el("p", { className: "hint", text: "This listing is missing details the model needs, so there is no estimate. That says nothing about whether the rent is fair." }),
      ]);
    }
    return el("div", {}, [
      el("div", { className: "section-label", text: "Estimated fair rent (80% range)" }),
      el("div", { className: "fair-rent-line" }, [
        el("span", { className: "range", text: money(listing.fair_rent_low) + " – " + money(listing.fair_rent_high) }),
        el("span", { className: "hint", text: "Middle of range about " + money(listing.fair_rent) + ". An estimate, not an appraisal." }),
      ]),
      rangeGraphic(listing),
      badge,
    ]);
  }

  function scoreSection(listing) {
    if (listing.match_score === null) {
      return el("div", {}, [
        el("div", { className: "section-label", text: "Match score" }),
        el("p", { className: "hint", text: "Add a budget, size, distance or amenity preference to get a match score." }),
      ]);
    }
    const lowConfidence = listing.match_confidence < LOW_CONFIDENCE;
    const rows = Object.entries(listing.score_components).map(([key, component]) =>
      el("tr", {}, [
        el("td", { text: component.label }),
        component.known
          ? el("td", { className: "num", text: percent(component.score) })
          : el("td", { className: "num unknown-cell", text: "not listed (counted as 50%)" }),
        el("td", { className: "num", text: percent(component.weight) }),
      ])
    );
    return el("div", {}, [
      el("div", { className: "section-label", text: "Match score" }),
      el("div", { className: "score-row" }, [
        el("span", { className: "score-number", text: percent(listing.match_score) }),
        el("span", { className: "confidence" + (lowConfidence ? " low" : ""), text: "confidence " + percent(listing.match_confidence) }),
      ]),
      el("div", { className: "meter", role: "img", "aria-label": "Match score " + percent(listing.match_score) }, [
        el("div", { className: "meter-fill", style: { width: percent(listing.match_score) } }),
      ]),
      lowConfidence
        ? el("p", { className: "hint", text: "Some of what you asked about is not stated in this listing. Those parts count as neutral, not as a miss." })
        : null,
      el("details", { className: "breakdown" }, [
        el("summary", { text: "Score breakdown" }),
        el("table", {}, [
          el("thead", {}, [el("tr", {}, [el("th", { text: "Part" }), el("th", { className: "num", text: "Score" }), el("th", { className: "num", text: "Weight" })])]),
          el("tbody", {}, rows),
        ]),
      ]),
    ]);
  }

  function petChip(listing) {
    if (listing.pets_allowed === null) {
      return el("span", { className: "chip unknown", title: "The listing does not say whether pets are allowed. This is not the same as no pets." }, [
        el("span", { "aria-hidden": "true", text: "?" }),
        "Pet policy not listed",
      ]);
    }
    if (listing.pets_allowed === "none_allowed") {
      return el("span", { className: "chip no" }, [el("span", { "aria-hidden": "true", text: "✕" }), "No pets allowed"]);
    }
    return el("span", { className: "chip yes" }, [el("span", { "aria-hidden": "true", text: "✓" }), listing.pet_policy]);
  }

  function amenityChips(listing, wanted) {
    if (listing.amenities === null) {
      return [el("span", { className: "chip unknown", title: "The listing has no amenity list. Amenities may still exist." }, [
        el("span", { "aria-hidden": "true", text: "?" }),
        "Amenities not listed",
      ])];
    }
    const available = new Set(listing.amenities.map((item) => item.toLowerCase()));
    const chips = wanted.map((item) =>
      available.has(item.toLowerCase())
        ? el("span", { className: "chip wanted" }, [el("span", { "aria-hidden": "true", text: "✓" }), item])
        : el("span", { className: "chip unknown", title: "Not mentioned in this listing's amenity list." }, [item + " not mentioned"])
    );
    const wantedSet = new Set(wanted.map((item) => item.toLowerCase()));
    const others = listing.amenities.filter((item) => !wantedSet.has(item.toLowerCase()));
    others.slice(0, 8).forEach((item) => chips.push(el("span", { className: "chip", text: item })));
    if (others.length > 8) {
      chips.push(el("span", { className: "chip", text: "+" + (others.length - 8) + " more" }));
    }
    return chips;
  }

  function describeSize(listing) {
    const parts = [];
    if (listing.bedrooms === null) {
      parts.push("bedrooms not listed");
    } else {
      parts.push(listing.bedrooms === 0 ? "Studio" : listing.bedrooms + " bd");
    }
    parts.push(listing.bathrooms === null ? "bathrooms not listed" : listing.bathrooms + " ba");
    if (listing.square_feet !== null) {
      parts.push(listing.square_feet.toLocaleString("en-US") + " sq ft");
    }
    return parts.join(" · ");
  }

  function describeDistance(listing) {
    if (listing.distance_from_campus_miles === null) {
      return "Location not listed, so distance from campus is unknown";
    }
    return "About " + Math.round(listing.distance_from_campus_miles) + " mi from campus (area-level location)";
  }

  function card(listing, wanted, petsRequested) {
    const unknown = listing.unknown_fields.map((field) => UNKNOWN_FIELD_LABELS[field] || field);
    const perBedroom = listing.bedrooms !== null ? listing.price / Math.max(listing.bedrooms, 1) : null;
    return el("article", { className: "card" }, [
      el("div", { className: "card-head" }, [
        el("div", {}, [
          el("h3", { className: "card-title", text: (listing.cityname || "City not listed") + ", " + listing.state }),
          el("div", { className: "card-sub", text: describeSize(listing) }),
          el("div", { className: "card-sub", text: describeDistance(listing) }),
        ]),
        el("div", { className: "asking" }, [
          el("div", { className: "section-label", text: "Asking rent (2018–19)" }),
          el("div", { className: "amount", text: money(listing.price) }),
          el("div", { className: "per", text: perBedroom !== null ? money(perBedroom) + " per bedroom" : "" }),
          el("div", { className: "per", text: listing.price_per_sqft !== null ? "$" + listing.price_per_sqft.toFixed(2) + " per sq ft" : "" }),
        ]),
      ]),
      el("div", { className: "card-grid" }, [fairRentSection(listing), scoreSection(listing)]),
      el("div", { className: "chips" }, [petChip(listing), ...amenityChips(listing, wanted)]),
      petsRequested && listing.pets_unconfirmed
        ? el("p", { className: "hint", text: "Shown below confirmed pet-friendly listings because the policy is not stated. Ask the landlord before ruling it in or out." })
        : null,
      unknown.length
        ? el("div", { className: "unknown-line", text: "Not stated in this listing: " + unknown.join(", ") + ". Unknown means not listed, not no." })
        : null,
      listing.data_flags.length
        ? el("div", { className: "flag-line", text: "Data check: " + listing.data_flags.join("; ") + "." })
        : null,
    ]);
  }

  async function search(event) {
    if (event) {
      event.preventDefault();
    }
    const params = buildQuery();
    window.history.replaceState(null, "", "?" + params.toString());
    resultsSummary.textContent = "Searching…";
    resultsList.replaceChildren();
    try {
      const response = await fetch("/api/search?" + params.toString());
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Search failed");
      }
      const wanted = params.get("amenities") ? params.get("amenities").split(",") : [];
      const petsRequested = Boolean(params.get("pets"));
      const unconfirmed = payload.results.filter((listing) => listing.pets_unconfirmed).length;
      let summary = "Showing " + payload.count.toLocaleString("en-US") + " of " + payload.total_matches.toLocaleString("en-US") + " matching listings, best match first.";
      if (petsRequested && unconfirmed) {
        summary += " " + unconfirmed + " shown do not state a pet policy and are listed after confirmed matches.";
      }
      resultsSummary.textContent = summary;
      if (!payload.results.length) {
        resultsList.appendChild(el("div", { className: "panel empty", text: "No listings match those filters. Try widening the distance or raising the maximum rent." }));
        return;
      }
      resultsList.replaceChildren(...payload.results.map((listing) => card(listing, wanted, petsRequested)));
    } catch (error) {
      resultsSummary.textContent = "";
      resultsList.appendChild(el("div", { className: "panel error", text: "Could not load listings: " + error.message }));
    }
  }

  async function loadDrivers() {
    const chart = document.getElementById("drivers-chart");
    const effects = document.getElementById("drivers-effects");
    try {
      const response = await fetch("/api/feature-importance");
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Unavailable");
      }
      const rows = payload.importance
        .filter((row) => row.forest_permutation_dollars > 0)
        .slice(0, DRIVER_COUNT);
      const maximum = Math.max(...rows.map((row) => row.forest_permutation_dollars), 1);
      chart.replaceChildren(
        ...rows.map((row) => {
          const tip = row.feature + ": scrambling it raises typical error by " + money(row.forest_permutation_dollars) +
            (row.linear_dollars !== null ? ". Linear model: " + signedMoney(row.linear_dollars) + " for " + row.linear_change : "");
          return el("div", { className: "driver-row", title: tip }, [
            el("span", { className: "driver-name", text: row.feature }),
            el("div", { className: "driver-track" }, [
              el("div", { className: "driver-bar", style: { width: (row.forest_permutation_dollars / maximum) * 100 + "%" } }),
            ]),
            el("span", { className: "driver-value", text: money(row.forest_permutation_dollars) }),
          ]);
        })
      );
      const topEffects = payload.linear_effects
        .slice()
        .sort((first, second) => Math.abs(second.dollars) - Math.abs(first.dollars))
        .slice(0, EFFECT_COUNT);
      effects.replaceChildren(...topEffects.map((effect) => el("li", { text: signedMoney(effect.dollars) + " — " + effect.change })));
    } catch (error) {
      chart.replaceChildren(el("p", { className: "hint", text: "Rent drivers are not available yet. Run python -m src.model to build them." }));
    }
  }

  function applyQuery(params) {
    let applied = false;
    for (const [key, value] of params.entries()) {
      if (key === "amenities") {
        const wanted = new Set(value.split(",").map((item) => item.trim().toLowerCase()));
        amenityOptions.querySelectorAll("input[name=amenity]").forEach((input) => {
          input.checked = wanted.has(input.value.toLowerCase());
        });
        applied = true;
      } else if (key === "priorities") {
        const order = value.split(",").map((item) => item.trim());
        priorities.sort((first, second) => {
          const firstIndex = order.indexOf(first.key);
          const secondIndex = order.indexOf(second.key);
          return (firstIndex < 0 ? order.length : firstIndex) - (secondIndex < 0 ? order.length : secondIndex);
        });
        renderPriorities();
        applied = true;
      } else if (form.elements[key]) {
        const field = form.elements[key];
        if (field.type === "checkbox") {
          field.checked = ["1", "true", "yes", "on"].includes(value.toLowerCase());
        } else {
          field.value = value;
        }
        applied = true;
      }
    }
    return applied;
  }

  async function estimateRent(event) {
    event.preventDefault();
    const target = document.getElementById("estimate-result");
    const params = new URLSearchParams();
    for (const [key, value] of new FormData(event.target).entries()) {
      if (String(value).trim() !== "") {
        params.set(key, String(value).trim());
      }
    }
    target.replaceChildren(el("p", { className: "hint", text: "Estimating…" }));
    try {
      const response = await fetch("/api/estimate?" + params.toString());
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Estimate failed");
      }
      const label = VALUE_LABELS[payload.value_label] || VALUE_LABELS.unknown;
      const children = [
        el("div", { className: "section-label", text: "Fair rent (" + percent(payload.interval_coverage) + " range)" }),
        el("div", { className: "fair-rent-line" }, [
          el("span", { className: "range", text: money(payload.fair_rent_low) + " – " + money(payload.fair_rent_high) }),
          el("span", { className: "hint", text: "Middle of range about " + money(payload.fair_rent) + ". " + payload.note }),
        ]),
      ];
      if (payload.price !== null) {
        children.push(rangeGraphic(payload));
        children.push(el("span", { className: "badge " + label.className }, [el("span", { "aria-hidden": "true", text: label.icon }), label.text]));
      }
      target.replaceChildren(...children);
    } catch (error) {
      target.replaceChildren(el("p", { className: "error", text: error.message }));
    }
  }

  async function loadOptions() {
    try {
      const response = await fetch("/api/options");
      const payload = await response.json();
      priorities = payload.priorities;
      renderPriorities();
      renderAmenities(payload.amenities);
      renderStates(payload.states);
      const estimateState = document.getElementById("estimate-state");
      payload.states.forEach((state) => estimateState.appendChild(el("option", { value: state, text: state, selected: state === "IL" })));
      if (applyQuery(new URLSearchParams(window.location.search))) {
        search();
      }
    } catch (error) {
      resultsSummary.textContent = "Could not load search options.";
    }
  }

  form.addEventListener("submit", search);
  document.getElementById("estimate-form").addEventListener("submit", estimateRent);
  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      resultsSummary.textContent = "";
      resultsList.replaceChildren(el("div", { className: "panel empty", text: "Set what you are looking for and press Find listings." }));
    }, 0);
  });

  resultsList.appendChild(el("div", { className: "panel empty", text: "Set what you are looking for and press Find listings." }));
  loadOptions();
  loadDrivers();
})();
