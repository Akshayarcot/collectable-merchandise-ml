"""
app.py
Interactive Web-based Valuation Dashboard & Demo Server
UE24CS352A - Machine Learning Mini-Project
Authors: AR Akshay Kumar (PES1UG24CS705) & Shreyas Gowda (PES1UG24CS704)

Features:
- Real-time Multi-Model Collectible Valuation Engine
- Novel Feature Inputs: Community Hype Score & Vaulted / Discontinued Status
- Condition-Adjusted Market Appraisal (Mint 10, Near Mint 9, Box Damaged, Out of Box)
- Visual Class Probability & Model Consensus Breakdown
- Built with Python standard library http.server (no external web framework needed!)
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset, PRICE_BIN_EDGES
from src.preprocessing import CollectablesPreprocessor
from src.kernels import CustomCollectableKernel
from src.models import (
    BaselineLinearSVM,
    CustomKernelSVM,
    VanillaDecisionTree,
    BaggedDecisionTree,
    RandomForestModel
)

PORT = 8080

BIN_LABELS = {
    1: {"name": "Bin 1 (Common Retail)", "range": "$14.00 - $38.13", "desc": "Standard mass release, high liquidity"},
    2: {"name": "Bin 2 (Modest Premium)", "range": "$38.13 - $104.87", "desc": "Retailer exclusive, retired common"},
    3: {"name": "Bin 3 (Collectible Tier)", "range": "$104.87 - $287.02", "desc": "Comic-Con exclusive, convention chase"},
    4: {"name": "Bin 4 (Grail Status)", "range": "$287.02 - $785.55", "desc": "High rarity, limited numbered run"},
    5: {"name": "Bin 5 (Ultra-Grail / Museum)", "range": "$785.55 - $2,150.00", "desc": "Top-tier holy grail, vaulted icon"}
}

CONDITION_RATES = {
    "Mint_10": 1.00,
    "Near_Mint_9": 0.85,
    "Damaged_Box": 0.60,
    "Loose_OOB": 0.35
}

# Global models cache
CACHED_PREPROCESSOR = None
CACHED_MODELS = None


def init_models():
    global CACHED_PREPROCESSOR, CACHED_MODELS
    if CACHED_MODELS is not None:
        return
    print("[*] Training and caching ML models for Web Dashboard...")
    df = load_dataset()
    preprocessor = CollectablesPreprocessor(include_novel_features=True, random_state=42)
    preprocessor.fit(df)
    
    X_ohe = preprocessor.transform_one_hot(df)
    x_cont, z_cat = preprocessor.transform_ordinal(df)
    y = df["price_bin"].values
    
    lin_svm = BaselineLinearSVM(C=1.0, random_state=42).fit(X_ohe, y)
    
    ck_calc = CustomCollectableKernel(
        marginal_probs=preprocessor.marginal_probs,
        weights=[1.5, 2.5, 1.0, 1.0, 3.0, 2.0],
        theta=0.45,
        alpha=0.5,
        categorical_cols=preprocessor.categorical_cols
    )
    custom_svm = CustomKernelSVM(kernel_calculator=ck_calc, C=1.5, random_state=42).fit((x_cont, z_cat), y)
    
    dt = VanillaDecisionTree(criterion="gini", random_state=42).fit(X_ohe, y)
    bag = BaggedDecisionTree(n_estimators=75, random_state=42).fit(X_ohe, y)
    rf = RandomForestModel(n_estimators=100, random_state=42).fit(X_ohe, y)
    
    CACHED_PREPROCESSOR = preprocessor
    CACHED_MODELS = {
        "Baseline Linear SVM": lin_svm,
        "Custom Kernel SVM (Our Proposed)": custom_svm,
        "Vanilla Decision Tree": dt,
        "Bagged Decision Tree (75 Trees)": bag,
        "Random Forest Benchmark": rf
    }
    print("[+] Models initialized successfully!")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Collectable Merchandise ML Valuation Dashboard</title>
    <style>
        :root {
            --bg: #090d16;
            --surface: #131b2e;
            --surface-hover: #1a253f;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.4);
            --accent: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #1e293b;
            --gold: #f59e0b;
            --green: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 24px 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid var(--border); padding-bottom: 18px; }
        .title h1 { font-size: 1.6rem; color: #fff; font-weight: 700; }
        .title p { color: var(--accent); font-size: 0.95rem; margin-top: 4px; }
        .badge { background: #1e293b; border: 1px solid var(--primary); color: #93c5fd; padding: 6px 14px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
        .grid { display: grid; grid-template-columns: 420px 1fr; gap: 24px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .card-header { font-size: 1.15rem; font-weight: 600; color: #fff; margin-bottom: 18px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px; display: flex; align-items: center; justify-content: space-between; }
        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
        select, input[type="number"], input[type="range"] {
            width: 100%; padding: 10px 12px; background: #0c1222; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 0.9rem;
        }
        select:focus, input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary-glow); }
        .slider-val { float: right; color: var(--accent); font-weight: 700; }
        .btn-calc {
            width: 100%; background: linear-gradient(135deg, #2563eb, #0891b2); color: white; padding: 12px; border: none; border-radius: 10px;
            font-size: 1rem; font-weight: 700; cursor: pointer; transition: 0.2s; margin-top: 8px;
        }
        .btn-calc:hover { opacity: 0.95; transform: translateY(-1px); box-shadow: 0 4px 15px var(--primary-glow); }
        .preset-btns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 18px; }
        .preset-btn {
            background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 8px; padding: 8px; font-size: 0.8rem;
            cursor: pointer; text-align: left; transition: 0.2s;
        }
        .preset-btn:hover { background: #334155; color: #fff; border-color: var(--accent); }
        .hero-result { background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(6,182,212,0.1)); border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; }
        .hero-tier { font-size: 1.8rem; font-weight: 800; color: #38bdf8; margin: 6px 0; }
        .hero-price { font-size: 1.4rem; color: var(--gold); font-weight: 700; }
        .hero-sub { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }
        .model-table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.88rem; }
        .model-table th { background: #0c1222; color: #94a3b8; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 1px solid var(--border); }
        .model-table td { padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .tier-pill { padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; display: inline-block; }
        .tier-1 { background: rgba(148,163,184,0.2); color: #cbd5e1; }
        .tier-2 { background: rgba(59,130,246,0.2); color: #93c5fd; }
        .tier-3 { background: rgba(16,185,129,0.2); color: #6ee7b7; }
        .tier-4 { background: rgba(245,158,11,0.2); color: #fcd34d; }
        .tier-5 { background: rgba(239,68,68,0.2); color: #fca5a5; }
        .tag-novel { font-size: 0.7rem; background: #064e3b; color: #6ee7b7; padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title">
                <h1>Collectable Merchandise Valuation Engine</h1>
                <p>UE24CS352A Machine Learning Mini-Project | Team AR Akshay Kumar & Shreyas Gowda</p>
            </div>
            <span class="badge">PES1UG24CS705 &amp; PES1UG24CS704</span>
        </header>

        <div class="grid">
            <!-- Input Form Card -->
            <div class="card">
                <div class="card-header">
                    <span>Collectible Attributes</span>
                    <span style="font-size: 0.75rem; color: var(--accent);">Live ML Model</span>
                </div>

                <label>Quick Evaluation Presets:</label>
                <div class="preset-btns">
                    <button class="preset-btn" onclick="applyPreset('beerus')">🔥 <b>Beerus SDCC</b><br><small>Ultra Rare Grail</small></button>
                    <button class="preset-btn" onclick="applyPreset('ironman')">⚡ <b>Iron Man Glow</b><br><small>Mid-Tier Exclusive</small></button>
                    <button class="preset-btn" onclick="applyPreset('simba')">🦁 <b>Simba Spirit</b><br><small>Common Retail</small></button>
                    <button class="preset-btn" onclick="applyPreset('tony')">🐯 <b>Tony Tiger Chase</b><br><small>Vaulted Grail Icon</small></button>
                </div>

                <form id="evalForm" onsubmit="event.preventDefault(); evaluateItem();">
                    <div class="form-group">
                        <label for="category">Fandom Category</label>
                        <select id="category">
                            <option value="Anime">Anime (DragonBall, Naruto, etc.)</option>
                            <option value="Marvel">Marvel (Avengers, Spider-Man)</option>
                            <option value="Ad-Icons">Ad-Icons (Tony Tiger, Chester)</option>
                            <option value="Disney">Disney (Mickey, Simba)</option>
                            <option value="Game-of-Thrones">Game of Thrones (Jon Snow)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="material">Material Finish</label>
                        <select id="material">
                            <option value="Standard/Vinyl">Standard / Vinyl</option>
                            <option value="Metallic">Metallic Finish</option>
                            <option value="Flocked">Flocked (Furry Texture)</option>
                            <option value="Glow-in-the-dark">Glow-in-the-dark</option>
                            <option value="Diamond Collection">Diamond Collection (Glitter)</option>
                            <option value="Chrome">Chrome Plated</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="version">Production / Convention Version</label>
                        <select id="version">
                            <option value="Standard">Standard Retail Release</option>
                            <option value="SDCC">San Diego Comic Con (SDCC)</option>
                            <option value="NYCC">New York Comic Con (NYCC)</option>
                            <option value="Chase">Chase Limited Variant (1:6 Ratio)</option>
                            <option value="Special Edition">Special Edition Sticker</option>
                            <option value="Vaulted">Vaulted / Discontinued</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Monthly Organic Search Volume: <span id="search_disp" class="slider-val">35,000</span></label>
                        <input type="range" id="search_volume" min="50" max="85000" step="500" value="35000" oninput="document.getElementById('search_disp').textContent = Number(this.value).toLocaleString()">
                    </div>

                    <!-- NOVEL FEATURES -->
                    <div class="form-group">
                        <label>Community Hype Score <span class="tag-novel">NOVEL ML FEATURE</span>: <span id="hype_disp" class="slider-val">8.5</span></label>
                        <input type="range" id="community_hype_score" min="0.0" max="10.0" step="0.1" value="8.5" oninput="document.getElementById('hype_disp').textContent = this.value">
                        <small style="color: #64748b; font-size: 0.72rem;">Reddit r/funkopop sentiment & discussion buzzword intensity</small>
                    </div>

                    <div class="form-group">
                        <label for="is_vaulted">Manufacturer Supply Status <span class="tag-novel">NOVEL ML FEATURE</span></label>
                        <select id="is_vaulted">
                            <option value="1">🔒 Vaulted / Discontinued (Supply Permanently Frozen)</option>
                            <option value="0">🏭 Active Production (Restocks Possible)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="box_condition">Box Physical Condition <span class="tag-novel">APPRAISAL ENGINE</span></label>
                        <select id="box_condition">
                            <option value="Mint_10">🌟 10/10 Mint in Box (100% Value)</option>
                            <option value="Near_Mint_9">✨ 9/10 Near Mint (Minor shelf wear, 85%)</option>
                            <option value="Damaged_Box">⚠️ 6/10 Box Damaged/Creased (60%)</option>
                            <option value="Loose_OOB">📦 Out of Box / Loose Figure (35%)</option>
                        </select>
                    </div>

                    <button type="submit" class="btn-calc">⚡ Calculate Live Valuation</button>
                </form>
            </div>

            <!-- Output Results Card -->
            <div class="card">
                <div class="card-header">
                    <span>Model Valuation & Appraisal Report</span>
                    <span id="calc_status" style="font-size: 0.75rem; color: var(--green);">● Models Ready</span>
                </div>

                <div class="hero-result" id="heroCard">
                    <div style="font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">Consensus Valuation</div>
                    <div class="hero-tier" id="resTier">Bin 4 (Grail Status)</div>
                    <div class="hero-price" id="resPrice">$287.02 - $785.55</div>
                    <div class="hero-sub" id="resCondition">Mint in Box Value | Condition Discount: 0%</div>
                </div>

                <h3 style="font-size: 1rem; color: #fff; margin-bottom: 10px;">Ensemble Model Predictions:</h3>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Predicted Tier</th>
                            <th>Estimated Value Range</th>
                        </tr>
                    </thead>
                    <tbody id="modelResults">
                        <!-- Populated by JS -->
                    </tbody>
                </table>

                <div style="margin-top: 25px; padding: 14px; background: #0c1222; border-radius: 8px; border: 1px solid var(--border);">
                    <h4 style="font-size: 0.85rem; color: var(--accent); margin-bottom: 6px;">💡 Technical Innovation Note:</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4;">
                        Our <b>Custom Kernel SVM</b> combines Gaussian RBF for search interest and community hype with Marco's probability-adjusted categorical overlap kernel. By weighting rare features (SDCC exclusivity, Vaulted status, and Hype score), it overcomes the blindness of linear models on scarce high-tier collectibles.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const PRESETS = {
            beerus: { category: "Anime", material: "Metallic", version: "SDCC", search_volume: 42000, hype: 9.4, vaulted: "1", condition: "Mint_10" },
            ironman: { category: "Marvel", material: "Glow-in-the-dark", version: "Special Edition", search_volume: 12500, hype: 6.8, vaulted: "1", condition: "Near_Mint_9" },
            simba: { category: "Disney", material: "Standard/Vinyl", version: "Standard", search_volume: 850, hype: 2.2, vaulted: "0", condition: "Mint_10" },
            tony: { category: "Ad-Icons", material: "Flocked", version: "Chase", search_volume: 28000, hype: 8.5, vaulted: "1", condition: "Mint_10" }
        };

        function applyPreset(key) {
            const p = PRESETS[key];
            if (!p) return;
            document.getElementById("category").value = p.category;
            document.getElementById("material").value = p.material;
            document.getElementById("version").value = p.version;
            document.getElementById("search_volume").value = p.search_volume;
            document.getElementById("search_disp").textContent = Number(p.search_volume).toLocaleString();
            document.getElementById("community_hype_score").value = p.hype;
            document.getElementById("hype_disp").textContent = p.hype;
            document.getElementById("is_vaulted").value = p.vaulted;
            document.getElementById("box_condition").value = p.condition;
            evaluateItem();
        }

        async function evaluateItem() {
            const payload = {
                category: document.getElementById("category").value,
                material: document.getElementById("material").value,
                special_version: document.getElementById("version").value,
                search_volume: Number(document.getElementById("search_volume").value),
                community_hype_score: Number(document.getElementById("community_hype_score").value),
                is_vaulted: Number(document.getElementById("is_vaulted").value),
                box_condition: document.getElementById("box_condition").value
            };

            document.getElementById("calc_status").textContent = "Calculating...";
            try {
                const res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                renderResults(data);
                document.getElementById("calc_status").textContent = "● Up to date";
            } catch (err) {
                console.error(err);
                document.getElementById("calc_status").textContent = "Error";
            }
        }

        function renderResults(data) {
            document.getElementById("resTier").textContent = data.consensus.name;
            document.getElementById("resPrice").textContent = data.consensus.display_price;
            document.getElementById("resCondition").textContent = data.consensus.condition_note;

            const tbody = document.getElementById("modelResults");
            tbody.innerHTML = "";
            for (const [modelName, info] of Object.entries(data.models)) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 600;">${modelName}</td>
                    <td><span class="tier-pill tier-${info.bin}">${info.name}</span></td>
                    <td style="color: #f59e0b; font-weight: 600;">${info.range}</td>
                `;
                tbody.appendChild(tr);
            }
        }

        window.onload = () => { evaluateItem(); };
    </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            
            init_models()
            
            sample_df = pd.DataFrame([{
                "category": data.get("category", "Anime"),
                "material": data.get("material", "Metallic"),
                "special_version": data.get("special_version", "SDCC"),
                "search_volume": data.get("search_volume", 35000),
                "community_hype_score": data.get("community_hype_score", 8.0),
                "is_vaulted": data.get("is_vaulted", 1)
            }])
            
            X_ohe = CACHED_PREPROCESSOR.transform_one_hot(sample_df)
            x_cont, z_cat = CACHED_PREPROCESSOR.transform_ordinal(sample_df)
            
            cond_key = data.get("box_condition", "Mint_10")
            cond_multiplier = CONDITION_RATES.get(cond_key, 1.0)
            
            model_outputs = {}
            bins = []
            
            for name, model in CACHED_MODELS.items():
                if "Custom Kernel" in name:
                    pred_bin = int(model.predict((x_cont, z_cat))[0])
                else:
                    pred_bin = int(model.predict(X_ohe)[0])
                bins.append(pred_bin)
                
                b_info = BIN_LABELS[pred_bin]
                low = PRICE_BIN_EDGES[pred_bin - 1] * cond_multiplier
                high = PRICE_BIN_EDGES[pred_bin] * cond_multiplier
                
                model_outputs[name] = {
                    "bin": pred_bin,
                    "name": b_info["name"],
                    "range": f"${low:.2f} - ${high:.2f}",
                    "desc": b_info["desc"]
                }
                
            consensus_bin = int(np.median(bins))
            c_info = BIN_LABELS[consensus_bin]
            low_c = PRICE_BIN_EDGES[consensus_bin - 1] * cond_multiplier
            high_c = PRICE_BIN_EDGES[consensus_bin] * cond_multiplier
            
            disc_pct = int((1.0 - cond_multiplier) * 100)
            note = f"Physical Condition: {cond_key.replace('_', ' ')} (Discount: {disc_pct}%)" if disc_pct > 0 else "10/10 Mint in Box (100% Value)"
            
            response = {
                "consensus": {
                    "bin": consensus_bin,
                    "name": c_info["name"],
                    "display_price": f"${low_c:.2f} - ${high_c:.2f}",
                    "condition_note": note
                },
                "models": model_outputs
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Quiet standard server logging
        return


def main():
    init_models()
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"\n" + "=" * 65)
    print(f"🚀 COLLECTABLE VALUATION WEB DASHBOARD READY!")
    print(f"🔗 Open in browser: http://localhost:{PORT}")
    print(f"=" * 65 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
