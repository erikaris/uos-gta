# Overview of Both Documents

Let me break this down clearly for you as a TA.

---

## 📋 What Is the Lab About?

The lab session ("Critiquing Visualisations") is a **practical seminar** where students learn how to look at data visualisations critically — not just "does it look nice?" but asking deeper questions like: *What story is this telling? Who is it for? What's misleading or missing?* It's a group-based, discussion-driven session with three core tasks that build on each other.

---

## 🗺️ Learning Flow / Outline of the Lab

The lab moves through three escalating levels of analysis:

**Level 1 — Understanding the Story** (Task 1)
Students look at a visualisation and ask: *what is actually being shown here, and what narrative is it trying to convey?*

**Level 2 — Understanding the Analytical Task** (Task 2)
Students step into the shoes of the *creator* and ask: *who uses this? why was it built? what problem does it solve?*

**Level 3 — Full Critique** (Task 3)
Students combine everything and evaluate a visualisation holistically — the good, the bad, and what they'd improve.

The session ends with a **coursework discussion**, which gives students context for their assignment (due 1st June 2026).

---

## ✅ Solving the Lab Tasks

### Task 1: Understanding the Story — Temperature Anomaly Chart

The chart shows **global temperature anomaly (°C) from 1880 to roughly 2015**, with four independent data sources (NOAA, NASA, Japanese Meteorological Agency, and the UK Met Office Hadley Centre) all plotted together.

**1. What is the visualisation showing?**
It shows how average global temperatures have deviated from the 1910–2000 baseline average over more than a century, using four separate scientific datasets that all agree with each other.

**2. What story is it trying to tell?**
The core story is: *global warming is real and consistent across independent scientific institutions*. The fact that four completely separate organisations using different methods arrive at near-identical results is the key rhetorical point — it's hard to dismiss.

- **Backstory:** This was produced in the context of climate change debate, where some people argue that temperature records are unreliable or biased. Using four independent sources counters that argument directly.
- **Generic or specialised?** It's accessible to a general audience but having some basic climate literacy helps — knowing what "temperature anomaly" means (deviation from a baseline, not the actual temperature) is important.
- **Would you need extra knowledge?** Yes — the concept of a "baseline average" (1910–2000 in this case) needs explaining. Without it, viewers might not understand why the y-axis goes negative.
- **Positives:** Clean design, multiple credible sources, clear upward trend is unmistakable, good use of colour to separate sources.
- **Negatives:** The four lines can overlap and become hard to distinguish; no explanation of what "temperature anomaly" means in plain language on the chart itself.

**Catchy title ideas:**
- *"Four Scientists Walk Into a Room — They All Agree: It's Getting Hotter"*
- *"No Matter Who Measures It, the World Is Warming"*
- *"The Uncomfortable Consensus: 135 Years of Rising Temperatures"*

---

### Task 1 (continued): NYC Street Trees Visualisation

This is an **interactive visualisation** from Cloudred showing the variety and quantity of street trees across the five boroughs of New York City (Bronx, Brooklyn, Manhattan, Queens, Staten Island) using NYC Open Data from 2005 and 2015.

**1. What is it showing?**
It shows the distribution of different tree species (by genus) across each borough, using colour-coded horizontal bands where the width of each colour segment represents the number of trees of that species.

**2. What story is it trying to tell?**
*New York City's urban forest is surprisingly rich and varied — and it's unevenly distributed across boroughs.* Queens has the most trees (254,860), while Manhattan has the fewest (51,661), which makes intuitive sense given population density and available space.

- **Backstory:** Urban trees are important for air quality, mental health, temperature regulation, and biodiversity. This is likely part of a broader green infrastructure or environmental policy context.
- **Generic or specialised?** Fairly accessible, but understanding why tree diversity matters (urban ecology) adds depth.
- **Extra knowledge needed?** Tree genus names (Oak, Maple, Elm, etc.) help — though the icons make it more accessible. Understanding borough geography also helps.
- **Positives:** Visually engaging, allows comparison across boroughs, interactive (clicking a species filters across boroughs), the tree silhouette icons are a clever touch.
- **Negatives:** The colour coding across many species is hard to track; proportional comparisons between boroughs are tricky because the bar lengths differ; the dark background, while stylish, may reduce accessibility.

---

### Task 2: Decision-Making Scenario — Autoimmune Diseases Chart

The visualisation from *Scientific American* titled **"The Terrible Toll of 76 Autoimmune Diseases"** shows rows of autoimmune disorders, each with three pieces of information displayed visually: the gender split of patients (percent female), the average age of onset, and the frequency in the population.

**1. Who could be a user of this dashboard?**
- **General public** — someone recently diagnosed wanting to understand how common their condition is
- **Patients and caregivers** — comparing their condition to others, understanding gender patterns
- **Medical students and junior doctors** — getting a broad overview of autoimmune disease epidemiology
- **Science journalists** — communicating health data to the public
- **Policy makers in health** — understanding disease burden across populations

**2. What is the primary purpose?**
To give a **comparative overview** of multiple autoimmune conditions simultaneously — letting users see patterns across diseases (e.g., most are more common in women; onset tends to cluster in certain age ranges) without needing to read separate articles for each condition.

**3. What problem is it solving?**
The problem is that autoimmune diseases are numerous, varied, and often poorly understood even by patients who have them. Most resources cover one disease at a time. This dashboard solves the **fragmentation problem** — it lets you see the whole landscape at once.

**4. How does the user solve this problem using the dashboard?**
A user scans down the rows to find their condition, then reads horizontally to understand: *Is this more common in my gender? When does it typically start? How rare or common is it?* They can also scan vertically to notice patterns — for example, noticing that almost all autoimmune conditions disproportionately affect women.

---

### Task 3: Critiquing a Visualisation

The lab offers three options. Here's a worked example for **"The World's Billionaire Population by Country"** (a common Tableau Public visualisation type), though the same framework applies to any of the three.

**Framework for critique (applicable to all three):**

| Dimension | Questions to ask |
|---|---|
| Decision-maker use | Who uses this and for what decision? |
| Positives | What works well visually and informationally? |
| Improvements | What is missing, misleading, or inaccessible? |

**Example critique — "Plenty More Fish in the Sea"** (a visualisation about global fish stocks):

- **How a decision maker uses it:** A fisheries policy maker or environmental journalist would use it to understand which fish populations are depleted, recovering, or stable — informing quotas or conservation campaigns.
- **What works:** If it uses a world map with colour coding, it gives immediate geographic intuition. If it shows change over time, it tells a narrative arc.
- **What to improve:** Fish stock data is complex — combining species, region, and time on one chart risks oversimplification. Ethical issues arise if the data source (e.g., national fishing bodies) has incentives to under-report depletion. Accessibility issues might include colour-only encoding (problematic for colour-blind viewers) and lack of data source transparency.

---

## 📚 Summary of the Lecture (Ethics in Data Visualisation)

Here's the lecture content broken down into digestible chunks:

### 1. What Is Ethics?
The lecture distinguishes between morality (personal), ethics (group/professional codes), and law (external enforcement). **Data ethics** is about how organisations collect, protect, and use data. **Visualisation ethics** specifically means presenting data honestly — not distorting, misleading, or misrepresenting it.

### 2. Why Does This Matter for Visualisation?
Charts feel objective. They carry an "air of authority" (Correll, 2018). People trust graphs in a way they don't trust plain text. This makes misleading visualisations *more* dangerous than misleading prose — the audience doesn't question them as readily.

Importantly, there is currently **no widely accepted ethical framework** specifically for data visualisation.

### 3. Case Study: Clinical Artery Visualisations
Researchers tested whether 2D or 3D representations of arteries, combined with either rainbow or diverging colour maps, affected doctors' diagnostic accuracy. Results: **2D + diverging colour maps led to 91% accuracy**, while 3D + rainbow maps dropped accuracy significantly. This shows that visualisation choices have real-world consequences — in this case, literally life-or-death ones.

### 4. Data Is Not Objective
A common myth is that data "speaks for itself." The lecture pushes back on this. While data is quantitative and repeatable, it is always collected, processed, and interpreted by humans — meaning human biases get baked in.

**Types of bias covered:**
- **Selection bias** — only certain people/groups are captured (e.g., Boston's pothole app disadvantaged neighbourhoods where residents were less likely to own smartphones)
- **Survivor bias** — only "survivors" of a process are in the dataset
- **Confirmation bias** — seeking data that confirms existing beliefs
- **Confounding variables** — a hidden third factor explains a relationship
- **Correlation ≠ Causation** — two things moving together doesn't mean one causes the other
- **Dunning-Kruger Effect, Normality Bias, Overfitting** — analytical pitfalls

**Policing example:** The lecture used a FiveThirtyEight visualisation to show how statistics can *appear* equitable (equal use-of-force rates among stopped people) while hiding systemic bias (more Black people being stopped in the first place). The statistic about stopped people is technically accurate but misleading without context.

### 5. Visualisations Are Not Neutral
The standard world map (Mercator projection) was used as an example. Africa appears smaller than it actually is because the cylindrical projection distorts areas away from the equator. Africa is actually 30.2 million km² — more than three times Canada's 9.1 million km² — but on a Mercator map they look comparable. This isn't just a technical quirk; it reflects and reinforces a Eurocentric worldview.

The lecture also cited 100 Colorado drivers following Google Maps into a muddy field — showing that people trust visualisations even when they contradict their physical reality.

### 6. Five Deceptive Design Patterns

| Pattern | What It Is | Example |
|---|---|---|
| **Size misrepresentation** | Using diameter instead of area to represent quantity | A circle looks 4x bigger when its diameter is doubled, not 2x |
| **Unnecessary 3D** | Adding 3D perspective that distorts angles | 3D pie charts make front slices look bigger than they are |
| **Truncated axes** | Starting the y-axis above zero to exaggerate differences | A small increase looks dramatic on a graph starting at 9,000 |
| **Going against expectations** | Reversing conventions without signalling it | Using green for losses and red for gains on a profit map |
| **Non-uniform colour maps** | Rainbow/jet colour maps create false visual boundaries | The "jet" colour map makes some values look dramatically different when they're not |

### 7. Hallmarks of Bad Visualisation
- **Inaccessibility** — using too many colours, tiny labels, or formats that exclude colour-blind users
- **Needless decoration** — visual elements that look impressive but add no information ("chartjunk")
- **Data misrepresentation** — graphical elements that don't faithfully reflect the numbers (e.g., the payday loans pie chart whose segments don't add up to 100%)
- **Lack of transparency** — no data source, no methodology, no context
- **Selective display** — showing only the data that supports a conclusion
- **Over-simplification** — losing important nuance to make a cleaner chart

### 8. Ethical Framework for Visualisation
Since no dedicated framework exists, the lecture borrows from the **Society of Professional Journalists (SPJ) Code of Ethics**, adapted for visualisation:

- **Seek truth** — use reliable, verifiable data sources; don't oversimplify
- **Minimise harm** — consider long-term implications of how data is presented
- **Act independently** — avoid conflicts of interest
- **Be accountable** — explain your choices and be open to criticism

---

## Key Takeaways as a TA:

When facilitating the lab, watch out for students who:
- Only find negatives in critique tasks (critique = balanced evaluation, not just fault-finding)
- Confuse the story the visualisation *tells* with what it *shows* — encourage them to think about the rhetorical purpose
- Struggle with Task 2 — prompt them to think about *who* would actually sit in front of this, and *what decision* they'd make
- In Task 3, push students to connect their critique back to the lecture concepts (bias, deceptive patterns, accessibility)
