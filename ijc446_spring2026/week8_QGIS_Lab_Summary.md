# QGIS Lab Worksheet – Detailed Step-by-Step Summary
**Module:** IJC446 Information Visualisation for Decision Making  
**Topic:** Visualising and Analysing Spatial Data using QGIS

---

## Background Context

- **QGIS** (Quantum GIS) = free, open-source, cross-platform GIS software.
- **Esri / ArcGIS** = commercial alternative (more polished but paid).
- QGIS works via **layers** – each layer adds a specific type of spatial information on top of the map canvas (e.g. streets, restaurants, street lights). Combining layers gives a richer picture of any geography.
- **Census Output Areas (OAs)** = the smallest unit used by ONS (UK) for aggregating census data. Higher levels: electoral wards → civil parishes → districts → counties.

---

## PART 0 – Installing QGIS

### Option 1: Latest Version (via OSGeo4W Installer)

1. **Download installer** → Go to [https://qgis.org/en/site/forusers/download.html](https://qgis.org/en/site/forusers/download.html) → Download the **OSGeo4W Network Installer** (64-bit for most modern computers).
2. **Run the installer** → Select **"Express Desktop Install"** → press **Next**.
3. **Choose download location** → Keep the default.
4. **Select packages** → Keep defaults (QGIS, GDAL, GRASS GIS are pre-checked).
5. **Agree to conditions** → Click through several agreement screens.
6. **Wait for installation** → Takes several minutes to download and install.

> ⚠️ Note: The instructor found this version froze on their machine. Option 2 is recommended for stability.

---

### Option 2: Older Stable Version (QGIS 2.18.28) – Recommended

1. **Download installer** → Go to [https://qgis.org/downloads/](https://qgis.org/downloads/) (Windows) or [https://qgis.org/downloads/macOS/](https://qgis.org/downloads/macOS/) (Mac).
2. **Find the file** `QGIS-OSGeo4W-2.18.28-2-Setup-x86_64.exe` → Scroll to approximately the middle of the page.
3. **Run and install** → Follow the same steps as Option 1.

> ⚠️ Screenshots in the worksheet are from version 2.18.28. Features exist in the latest version too, but the UI may look slightly different.

---

### Lab Computer (University)

1. **Check if already installed** → Click the Windows button (bottom-left) → search for **"QGIS"**.
2. **If not installed** → Open **"Software Centre"** → search for QGIS → install normally.

---

## PART 1 – Understanding the QGIS Interface

1. **System Menu** → Located at the very top of the window. Standard menu bar with File, Layer, View, Plugins, etc.
2. **Toolbars** → Below the menu bar. More toolbars can be enabled via **View → Toolbars**.
3. **Layers Panel** → On the left-hand side. Lists all currently loaded layers with visibility toggle (checkbox next to each layer name). Click once to hide, click again to show.
4. **Data Sources Browser** → Also on the left side. Lists available data source connections (files, databases, WMS servers, etc.).
5. **Map Canvas** → The large central area where the map is rendered and displayed.
6. **On-The-Fly (OTF) Projection** → Bottom-right corner. Shows the current coordinate reference system (CRS). QGIS auto-selects projection when data is imported. Different projections cause the map to appear at different angles (e.g., EPSG:4326 vs. Google/Bing's Web Mercator).

> 📖 More GUI detail: [https://docs.qgis.org/2.14/en/docs/user_manual/introduction/qgis_gui.html](https://docs.qgis.org/2.14/en/docs/user_manual/introduction/qgis_gui.html)

---

## EXERCISE 1 – Understanding Occupations/Health in Sheffield

### STEP 1 – Download Sheffield Boundary Data (OA Shapefile)

1. **Go to the boundary data portal** → [https://borders.ukdataservice.ac.uk/bds.html](https://borders.ukdataservice.ac.uk/bds.html)
2. **Make selections:**
   - Country: **England**
   - Type: **Census**
   - Period: **2011 and later**
   - Click **"Find"**
3. **Select boundary type** → From the Boundaries list, choose **"English Census Output Areas with OAC, 2011"**.
4. **Select area** → Click **"List Areas"** → Select **"Sheffield"** from the list → Click **"Expand Selection"** if needed.
5. **Extract data** → Click **"Extract Boundary Data"**.
6. **Download** → On the next page, confirm:
   - Data Format: **ESRI Shape File**
   - Archive Method: **Zip**
   - Click **BoundaryData.zip** to download.
7. **Unzip** → Extract the ZIP. You will get several files:
   - `england_oac_2011.dbf` – attribute data
   - `england_oac_2011.prj` – projection info
   - `england_oac_2011.shp` – **the main shapefile** ← use this
   - `england_oac_2011.shx` – spatial index
   - `README`, `TermsAndConditions`

---

### STEP 2 – Load the Shapefile into QGIS

1. **Open QGIS**.
2. **Add the layer** → Click **Layer** (top menu) → **Add Layer** → **Add Vector Layer**.
3. **Configure source** → Source type: leave as **"File"** → Click **"Browse"** → Navigate to and select `england_oac_2011.shp` → Click **"Open"**.
4. **View map** → Sheffield's Output Areas will appear on the map canvas as a green filled polygon map.
5. **Toggle layer visibility** → In the Layers panel, click the checkbox next to `england_oac_2011` to hide/show it.

> 📌 Note: The map may look tilted compared to Google Maps. This is due to the projection (the shapefile uses a British national grid projection, not Web Mercator).

---

### STEP 3 – Download Census Health Data from NOMIS

1. **Go to NOMIS** → [https://www.nomisweb.co.uk/](https://www.nomisweb.co.uk/)
2. **Access 2011 Census tables** → On the right side under "Census statistics", click **"2011 Search by topic (table finder)"**.
3. **Filter by topic** → On the left panel, check **"Health (general)"** → At the top, select the radio button for **"Output Area"**.
4. **Select table** → From results, click **"General health [QS302EW]"**.
5. **Choose geography** → On the next page, set "Type of area" to **"output areas 2011 in Yorkshire and The Humber"**.
6. **Download** → Click **"Download"** → A CSV file named `bulk` will start downloading → **Rename it to `general health`** before saving.
7. **Open the file** → Open `general health.csv` in **MS Excel** (or a spreadsheet app).

> The file contains columns: date, geography, geography code, Rural Urban, General Health All categories, Very good, Good, Fair, Bad, Very bad.

---

### STEP 4 – Calculate Health Percentages in Excel

1. **Create GoodHealth_percent column** → In a new column (e.g., column K, row 2), enter the formula:
   ```
   K2 = (F2 + G2) / E2
   ```
   (Very Good + Good) ÷ Total. Result should be approximately **0.69**.
2. **Copy formula down** → Drag the formula down for all rows.
3. **Create PoorHealth_percent column** → In the next column (e.g., column L), enter:
   ```
   L2 = (I2 + J2) / E2
   ```
   (Bad + Very Bad) ÷ Total.
4. **Copy formula down** → Drag down for all rows.
5. **Save the file** → Save as `general health.csv` (keep CSV format).

---

### STEP 5 – Import the CSV into QGIS

1. **Add delimited text layer** → In QGIS, click **Layer** → **Add Layer** → **Add Delimited Text Layer**.
2. **Configure import:**
   - Browse and select `general health.csv`
   - File format: select **CSV (comma separated values)**
   - Check **"First record has field names"**
   - Geometry definition: select **"No geometry (attribute only table)"**
3. **Click OK** → A new layer called `general health` appears in the Layers panel (as a table, not a map layer).

---

### STEP 6 – Join the CSV to the Shapefile (by OA Code)

1. **Open layer properties** → Double-click on `england_oac_2011` in the Layers panel.
2. **Go to Joins tab** → Click on the **"Joins"** section in the left menu of the Layer Properties window.
3. **Add a new join** → Click the **"+"** button at the bottom.
4. **Configure the join:**
   - **Join layer:** `general health`
   - **Join field:** `geography code`
   - **Target field:** `code`
   - Tick **"Cache join layer in virtual memory"**
5. **Click OK** → Click **OK** again to close Layer Properties.

> This links the CSV data to the corresponding OA polygon in the shapefile by matching OA codes.

---

### STEP 7 – Colour-Code OAs by Health (Graduated Choropleth)

1. **Open Layer Styling panel** → Click **View** → **Panel** → **Layer Styling** (enable if not already visible).
2. **Change renderer** → In the Layer Styling panel, click the dropdown that says **"Single symbol"** → change to **"Graduated"**.
3. **Select column** → Set the **Column** to `general health_GoodHealth_percent` (or similar joined field name).
4. **Classify** → Go to the **"Classes"** tab → Click **"Classify"** → QGIS automatically creates 5 classes.
5. **View map** → The OAs are now colour-coded (dark = lower good health, light = higher good health).
6. **Try Poor Health view** → Change the Column to `general health_PoorHealth_percent` → Click **Classify** again to reclassify.
7. **Try Very Bad Health** → Change Column to `general health_Very bad health` → Click **Classify**.

> 📌 Note: This dataset is from 2011 (collected every 10 years). It gives a historical indication, not current data. City centre OAs showed higher proportions of poor health.

---

### STEP 8 – Download and Visualise Mean Age Data

1. **Go back to NOMIS** → Search for **"Mean age"** → Select **"Age structure [KS102EW]"**.
2. **Choose geography** → Select **"output areas 2011 in Yorkshire and The Humber"** → Click **Download**.
3. **Save and open** → Rename the CSV to `mean age` → Open in MS Excel. The second-to-last column contains **Mean Age**.
4. **Import to QGIS** → Same process as Step 5: **Layer → Add Layer → Add Delimited Text Layer** → Select `mean age.csv` → CSV format → First record has field names → No geometry.
5. **Join to shapefile** → Same process as Step 6: Double-click `england_oac_2011` → Joins → "+" → Join layer: `mean age`, Join field: `geography code`, Target field: `code`.
6. **Visualise** → In Layer Styling: Graduated → Column: select mean age column (second-to-last field) → Classify → Change Color ramp to **"Greens"** to differentiate from the health map.

> Result: Older average ages appear around the city outskirts. City centre is younger (students/professionals). This aligns with expected demographic patterns.

---

### STEP 9 – Download and Visualise Economic Activity Data

1. **Go back to NOMIS** → Select topic: **"Economic activity"** → Select **"Economic activity [KS601UK]"**.
2. **Choose geography** → **"output areas 2011 in Yorkshire and The Humber"** → Download CSV.
3. **Rename and open** → Save as `economic activity` → Open in Excel. Key columns: **Economically Active** and **Economically Inactive** (further split into retired, part-time, long-term unemployed, etc.).
4. **Import to QGIS** → Same import process as above (Layer → Delimited Text → CSV → No geometry).
5. **Join to shapefile** → Same join process as above (geography code → code).
6. **Visualise** → In Layer Styling: Graduated → Select **Economically Active** or **Economically Inactive** column → Classify → view distribution across Sheffield OAs.

---

## EXERCISE 2 – Creating Cartograms

> ⚠️ Warning: This exercise took ~1.5 hours for the instructor. Consider completing Exercise 3 first.

### STEP 1 – Save the Project First

- Click **Project** → **Save** → Give it a memorable name.

---

### STEP 2 – Install the Cartogram Plugin

1. **Open Plugin Manager** → Click **Plugins** → **Manage and Install Plugins**.
2. **Search** → Type **"Cartogram"** in the search box.
3. **Install** → Select the Cartogram plugin → Click **"Install Plugin"** → Click **"Close"**.

---

### STEP 3 – Save the Shapefile with Joined Data

1. **Hide the original layer** → In the Layers panel, toggle visibility off for `england_oac_2011`.
2. **Save as new shapefile** → Right-click `england_oac_2011` → Select **"Save As"** → Name it **"SheffieldOA"** → Click OK.
3. **New layer appears** → `SheffieldOA` is added to the map (shows Sheffield without colour-coding).
4. **View attribute table** → Right-click `SheffieldOA` → **"Open Attribute Table"** to verify joined fields are embedded (note: field names may have been truncated/renamed during the save).

---

### STEP 4 – Generate the Cartogram

1. **Open Cartogram tool** → Click **Vector** → **Cartogram** → **"Create Cartogram"**.
2. **Configure:**
   - **Input layer:** `SheffieldOA`
   - **Area field:** Select the **mean age** field (named something like `mean age 17`)
   - **Number of iterations:** Keep at **5**
3. **Click OK** → Wait. This will take a long time due to the large number of OAs.
4. **Result** → A new layer called `Cartogram` is added. It will appear heavily distorted to represent variations in mean age across areas.

---

### STEP 5 – Colour-Code the Cartogram

1. **Hide SheffieldOA** → Toggle off visibility for `SheffieldOA` to see the cartogram clearly.
2. **Apply graduated colours** → In Layer Styling: change to **Graduated** → Select the **mean age** field → Click **Classify**.
3. **Interpret** → Light blue areas (younger populations) appear shrunken. Dark blue areas (older populations) appear expanded. This visually encodes both age and geographic distribution simultaneously.

---

## EXERCISE 3 – Importing Citizen-Generated Data (OpenStreetMap)

> This section uses the QuickOSM plugin to pull crowdsourced map data directly from OpenStreetMap.

### STEP 1 – Save Project and Prepare Canvas

1. **Save project** → **Project → Save**.
2. **Hide all layers** → Toggle off visibility for `england_oac_2011`, `SheffieldOA`, and `Cartogram` in the Layers panel (do NOT delete them).

---

### STEP 2 – Install QuickOSM Plugin

1. **Open Plugin Manager** → **Plugins → Manage and Install Plugins**.
2. **Search** → Type **"QuickOSM"** → Select the plugin.
3. **Install** → Click **"Install Plugin"** → Close.

---

### STEP 3 – Download Amenity Data from OpenStreetMap

1. **Open QuickOSM** → Click **Vector → QuickOSM → QuickOSM**.
2. **Configure query:**
   - Left panel: click **"Quick query"**
   - **Key:** `amenity`
   - **Value:** (leave blank to get all amenity types)
   - Select radio button **"In"** → Type **"Sheffield"**
3. **Click "Run Query"** → Wait for download to complete → Close the window.
4. **Result** → Several new layers appear in the Layers panel. Points are visible on the map showing all amenities.

---

### STEP 4 – Categorise Amenities by Type

1. **Select the point layer** → Click on `amenity_Sheffield` (the points layer) in the Layers panel.
2. **Change styling** → In Layer Styling panel: change renderer from **"Single symbol"** to **"Categorized"**.
3. **Set column** → Set Column to **"amenity"** → Click **"Classify"**.
4. **Result** → All amenity types (library, school, pub, pharmacy, hospital, clinic, etc.) are colour-coded on the map.

---

### STEP 5 – Filter for Health-Related Amenities Only

1. **Open Filter dialog** → Right-click on `amenity_Sheffield` in the Layers panel → Click **"Filter"**.
2. **Write filter expression** → In the "Provider specific filter expression" box, type:
   ```
   "amenity"='hospital' OR "amenity"='doctors' OR "amenity"='clinic'
   ```
3. **Test** → Click **"Test"** → A popup should say e.g. "The where clause returned 36 row(s)."
4. **Apply** → Click **OK** → Click **OK** again.
5. **Result** → Only hospitals, doctors, and clinics are shown as colour-coded points on the map.

---

### STEP 6 – Add Parking Data to the Filter

1. **Edit filter** → Right-click `amenity_Sheffield` → **"Filter"** → Modify expression to:
   ```
   "amenity"='hospital' OR "amenity"='doctors' OR "amenity"='clinic' OR "amenity"='parking'
   ```
2. **Click OK** → Parking spaces now also appear as points (shown in purple).

---

### STEP 7 – Download Bus Stop Data from OpenStreetMap

1. **Open QuickOSM again** → **Vector → QuickOSM → QuickOSM**.
2. **Configure new query:**
   - **Key:** `highway`
   - **Value:** `bus_stop`
   - Select **"In"** → Type **"Sheffield"**
3. **Run Query** → Wait for completion → Close.
4. **Result** → New layer `highway_bus_stop_Sheffield` is added with red dots for bus stops.
5. **Change colour** → In Layer Styling: change colour to **dark grey**.
6. **Reduce point size** → Click on the `highway_bus_stop` layer → Reduce point size to **0.8** to declutter the map.

---

### STEP 8 – Overlay Health Data with Amenity Points

1. **Make england_oac_2011 visible** → Toggle it on in the Layers panel.
2. **Change choropleth to poor health** → In Layer Styling: Graduated → Column: `general health_PoorHealth_percent` → Change Color ramp to **"YlGn"** (Yellow-Green) → Click **Classify**.
3. **Interpret combined map:**
   - Dark green OAs = higher proportion of poor health
   - Light blue/cyan points = health services (hospitals, clinics, doctors)
   - Purple points = parking spaces
   - Dark grey small dots = bus stops
4. **Identify gaps** → Zoom around to find dark green areas (poor health) that have few or no health service points nearby. These are candidate locations for a new clinic or hospital.

---

### STEP 9 – Add Google Maps as Base Layer

1. **Open XYZ tile connection** → In the Data Sources Browser panel on the left, find **"Tile Server XYZ"** → Right-click → **"New Connection"**.
2. **Enter URL** → In the popup, enter:
   ```
   https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}
   ```
   → Click **OK**.
3. **Name it** → In the next screen, enter **"GoogleMaps"** → Click **OK**.
4. **Add to map** → Double-click **"GoogleMaps"** in the XYZ Tile Server list → It's added as a new layer.
5. **Reorder layer** → If GoogleMaps covers everything, drag it below `england_oac_2011` in the Layers panel.
6. **Set transparency** → Click `england_oac_2011` → In Layer Styling, scroll to **"Layer rendering"** → Set **Layer transparency** to about **50%**.
7. **Result** → The choropleth is now semi-transparent over Google Maps, showing street names and neighbourhoods beneath the health data, making it easier to identify locations geographically.

---

## Key Concepts Summary

| Concept | Explanation |
|---|---|
| **Layers** | Different datasets stacked on top of each other on the map canvas |
| **Shapefile (.shp)** | ESRI file format storing geographic boundary data |
| **Output Area (OA)** | Smallest ONS census geography unit |
| **Choropleth** | Map where areas are shaded by a data variable (graduated colours) |
| **Cartogram** | Map where area shapes are distorted to represent a data variable |
| **Join** | Linking a CSV data table to a shapefile via a shared field (OA code) |
| **QuickOSM** | QGIS plugin to download OpenStreetMap data directly |
| **OTF Projection** | QGIS automatically assigns a projection when data is loaded |
| **Filter/Query** | SQL-like expression to show only specific features (e.g. only hospitals) |

---

## Final Reflective Task (Open-Ended)

The worksheet closes by asking students to think beyond the exercise:

- Can you identify other areas in Sheffield that need health services by overlaying additional data?
- Could you perform a similar analysis to identify where a **new school** should be built?
- What Census variables would help in that decision (e.g. age of population, number of children, current schools from OSM)?
- How could more OSM layers (schools, parks, transport) enrich the decision-making analysis?

---

*End of Summary*
