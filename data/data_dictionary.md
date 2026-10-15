# Collectable Merchandise Dataset Schema
**Dataset:** `data/collectables_dataset.csv`  
**Total Records:** 358 items  
**Domain:** Pop-Culture Collectibles (Funko Pop! Figurines) Secondary Market

---

### Feature Dictionary

| Column Name | Data Type | Description | Values / Examples |
| :--- | :--- | :--- | :--- |
| `item_id` | String | Unique identifier for each collectible | `POP-0001` to `POP-0358` |
| `item_name` | String | Commercial product title including character, finish, and series number | e.g., *"Funko Pop! Animation DragonBall Z Beerus (Metallic) SDCC Figure #120"* |
| `category` | Categorical | Fandom genre category (5 classes) | `Anime` (198), `Marvel` (69), `Ad-Icons` (43), `Disney` (37), `Game-of-Thrones` (11) |
| `topic` | String | Target character or franchise subject | e.g., *DragonBall Z Beerus, Iron Man, Mickey Mouse* |
| `material` | Categorical | Physical finish/material variant | `Standard/Vinyl`, `Metallic`, `Flocked`, `Glow-in-the-dark`, `Diamond Collection`, `Chrome` |
| `special_version` | Categorical | Production edition / convention exclusivity | `Standard`, `SDCC` (San Diego Comic Con), `NYCC` (New York Comic Con), `Chase`, `Special Edition`, `Vaulted` |
| `search_volume` | Integer | Normalized monthly organic search interest (proxy for demand) | Range: 50 – 85,000 monthly queries |
| `community_hype_score` | Float | **(Novel Feature)** Reddit `r/funkopop` collector hype & discussion intensity | Range: 0.0 – 10.0 |
| `is_vaulted` | Integer | **(Novel Feature)** Manufacturer production status (0 = Active, 1 = Discontinued/Vaulted) | 0 or 1 |
| `average_sale_price` | Float | Secondary market transaction price in USD ($) | Range: $14.00 – $2,150.00 |
| `price_bin` | Integer | Target classification label (5 log-spaced intervals) | 1, 2, 3, 4, 5 |

---

### Target Price Bins

| Price Bin | Price Interval ($) | Count | Market Tier Description |
| :---: | :---: | :---: | :--- |
| **Bin 1** | `[$14.00, $38.13]` | 146 | **Common Retail:** Mass production, standard distribution. |
| **Bin 2** | `($38.13, $104.87]` | 153 | **Modest Premium:** Retired common, minor retailer exclusive. |
| **Bin 3** | `($104.87, $287.02]` | 46 | **Collectible / Con Exclusive:** NYCC, standard Chase, high-demand series. |
| **Bin 4** | `($287.02, $785.55]` | 10 | **Grail Tier:** SDCC limited run, rare vaulted icons, flocked/metallic variants. |
| **Bin 5** | `($785.55, $2150.00]` | 3 | **Ultra-Rare Museum Tier:** Holy grails, numbered convention pieces. |
