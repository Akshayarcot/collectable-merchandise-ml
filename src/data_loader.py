"""
data_loader.py
Dataset generation, validation, and ingestion for Collectable Merchandise Evaluation.
Matches the exact empirical distribution of the Stanford CS229 Funko Pop Collectibles study:
- Categories: Anime (198), Marvel (69), Ad-Icons (43), Disney (37), Game-of-Thrones (11) = Total 358 items.
- 5 Price Bins:
    Bin 1: [14.00, 38.13]   (146 items)
    Bin 2: (38.13, 104.87]  (153 items)
    Bin 3: (104.87, 287.02] (46 items)
    Bin 4: (287.02, 785.55] (10 items)
    Bin 5: (785.55, 2150.00](3 items)
"""

import os
import numpy as np
import pandas as pd

PRICE_BIN_EDGES = [14.00, 38.13, 104.87, 287.02, 785.55, 2150.00]
PRICE_BIN_LABELS = [1, 2, 3, 4, 5]

CATEGORY_COUNTS = {
    "Anime": 198,
    "Marvel": 69,
    "Ad-Icons": 43,
    "Disney": 37,
    "Game-of-Thrones": 11
}

SAMPLE_TOPICS = {
    "Anime": [
        "DragonBall Z Beerus", "Naruto Shippuden Kurama", "Attack on Titan Eren Jaeger",
        "One Piece Monkey D Luffy", "My Hero Academia All Might", "Demon Slayer Tanjiro",
        "Death Note Ryuk", "Hunter x Hunter Killua", "Bleach Ichigo Kurosaki",
        "Jujutsu Kaisen Gojo Satoru", "DragonBall Super Goku Black", "Sailor Moon Moon Princess",
        "Fullmetal Alchemist Edward Elric", "Cowboy Bebop Spike Spiegel", "Yu-Gi-Oh Blue-Eyes White Dragon"
    ],
    "Marvel": [
        "Avengers Endgame Iron Man", "Spider-Man Far From Home", "Black Panther Wakanda Forever",
        "Thor Ragnarok Glow", "Captain America With Mjolnir", "Loki TVA Variant",
        "Deadpool Chimichanga Truck", "Doctor Strange Eye of Agamotto", "Wolverine Weapon X",
        "Thanos Infinity Gauntlet", "Scarlet Witch Chaos Magic", "Venom Carnage Symbiote"
    ],
    "Ad-Icons": [
        "Kelloggs Tony the Tiger", "McDonalds Ronald McDonald", "KFC Colonel Sanders",
        "Pillsbury Doughboy", "Energizer Bunny", "Kool-Aid Man",
        "Cheetos Chester Cheetah", "Pringles Julius Pringle", "Planters Mr Peanut"
    ],
    "Disney": [
        "Mickey Mouse Steamboat Willie", "Lion King Simba Spirit", "Beauty and the Beast Belle",
        "Frozen Elsa Ice Palace", "Aladdin Genie with Lamp", "Nightmare Before Christmas Jack Skellington",
        "Toy Story Buzz Lightyear", "Little Mermaid Ariel Glitter", "Pinocchio Jiminy Cricket"
    ],
    "Game-of-Thrones": [
        "Jon Snow King in North", "Daenerys Targaryen with Drogon", "Night King on Viserion",
        "Tyrion Lannister Hand of Queen", "Arya Stark Needle", "Brienne of Tarth Armor"
    ]
}

MATERIALS = ["Standard/Vinyl", "Metallic", "Flocked", "Glow-in-the-dark", "Diamond Collection", "Chrome"]
SPECIAL_VERSIONS = ["Standard", "SDCC", "NYCC", "Chase", "Special Edition", "Vaulted"]


def generate_curated_dataset(random_state: int = 42) -> pd.DataFrame:
    """
    Synthesizes the curated secondary-market collectibles dataset
    strictly adhering to the Stanford CS229 paper's distributional properties.
    """
    rng = np.random.RandomState(random_state)
    
    # Target bin distribution matching empirical paper
    # Bins 1: 146, 2: 153, 3: 46, 4: 10, 5: 3 (Total = 358)
    bin_targets = [146, 153, 46, 10, 3]
    
    # Sample prices strictly within each bin's bounds with log-uniform distribution
    all_prices = []
    all_bins = []
    
    for b_idx, count in enumerate(bin_targets):
        low = PRICE_BIN_EDGES[b_idx]
        high = PRICE_BIN_EDGES[b_idx + 1]
        log_low = np.log(low)
        log_high = np.log(high)
        sampled_log_prices = rng.uniform(log_low, log_high, size=count)
        sampled_prices = np.round(np.exp(sampled_log_prices), 2)
        all_prices.extend(sampled_prices)
        all_bins.extend([b_idx + 1] * count)
        
    prices = np.array(all_prices)
    bins = np.array(all_bins)
    
    # Randomly shuffle indices while keeping price-bin alignment
    perm = rng.permutation(len(prices))
    prices = prices[perm]
    bins = bins[perm]
    
    # Assign categories matching exact target counts
    cat_list = []
    for cat, count in CATEGORY_COUNTS.items():
        cat_list.extend([cat] * count)
    rng.shuffle(cat_list)
    
    # Sort or align so that high bins correlate naturally with rare features and high search volume
    data_rows = []
    for i in range(len(prices)):
        price = prices[i]
        bin_label = bins[i]
        category = cat_list[i]
        topic = rng.choice(SAMPLE_TOPICS[category])
        
        # Rare materials & special versions correlate with higher price bins
        if bin_label == 5:
            material = rng.choice(["Metallic", "Flocked", "Chrome"], p=[0.5, 0.3, 0.2])
            version = rng.choice(["SDCC", "Chase", "Vaulted"], p=[0.6, 0.3, 0.1])
            search_vol_base = rng.lognormal(mean=9.5, sigma=0.5)
        elif bin_label == 4:
            material = rng.choice(["Metallic", "Glow-in-the-dark", "Diamond Collection", "Standard/Vinyl"], p=[0.4, 0.3, 0.2, 0.1])
            version = rng.choice(["SDCC", "NYCC", "Chase", "Special Edition"], p=[0.4, 0.3, 0.2, 0.1])
            search_vol_base = rng.lognormal(mean=8.5, sigma=0.6)
        elif bin_label == 3:
            material = rng.choice(["Standard/Vinyl", "Glow-in-the-dark", "Metallic", "Diamond Collection"], p=[0.35, 0.25, 0.25, 0.15])
            version = rng.choice(["Standard", "NYCC", "Special Edition", "SDCC"], p=[0.3, 0.3, 0.25, 0.15])
            search_vol_base = rng.lognormal(mean=7.5, sigma=0.7)
        elif bin_label == 2:
            material = rng.choice(["Standard/Vinyl", "Metallic", "Glow-in-the-dark"], p=[0.65, 0.20, 0.15])
            version = rng.choice(["Standard", "Special Edition", "NYCC"], p=[0.6, 0.3, 0.1])
            search_vol_base = rng.lognormal(mean=6.5, sigma=0.8)
        else:  # bin 1
            material = rng.choice(["Standard/Vinyl", "Metallic"], p=[0.9, 0.1])
            version = rng.choice(["Standard", "Special Edition"], p=[0.85, 0.15])
            search_vol_base = rng.lognormal(mean=5.5, sigma=0.8)
            
        search_volume = int(np.clip(search_vol_base, 50, 85000))
        
        # Compose realistic item name
        item_id = f"POP-{i+1:04d}"
        if version != "Standard":
            item_name = f"Funko Pop! {category} {topic} ({material}) {version} Figure #{rng.randint(10, 999)}"
        else:
            item_name = f"Funko Pop! {category} {topic} ({material}) Figure #{rng.randint(10, 999)}"
            
        data_rows.append({
            "item_id": item_id,
            "item_name": item_name,
            "category": category,
            "topic": topic,
            "material": material,
            "special_version": version,
            "search_volume": search_volume,
            "average_sale_price": price,
            "price_bin": bin_label
        })
        
    df = pd.DataFrame(data_rows)
    return df


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Loads existing CSV dataset or generates and saves one if not present.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "collectables_dataset.csv")
        
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_curated_dataset()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset loaded successfully.")
    print(f"Total Items: {len(df)}")
    print("\nCategory Distribution:\n", df['category'].value_counts())
    print("\nPrice Bin Distribution:\n", df['price_bin'].value_counts().sort_index())
    print("\nSample Rows:\n", df[['item_name', 'category', 'material', 'special_version', 'search_volume', 'average_sale_price', 'price_bin']].head())
