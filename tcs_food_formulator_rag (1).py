# ============================================================
# TCS FOOD FORMULATOR - COMPLETE CORRECTED RAG PIPELINE
# ============================================================
# LIBRARIES TO INSTALL:
#   pip install langchain langchain-groq langchain-pinecone
#              langchain-huggingface langgraph pinecone-client
#              sentence-transformers pandas numpy python-dotenv
#              requests tqdm
# ============================================================

# ============================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================================
import os
import sys
import re
import json
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# UTF-8 safe printing for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', errors='replace').decode(
            sys.stdout.encoding or 'utf-8', errors='replace'))

# LangChain / LangGraph imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ── Load .env ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV    = os.getenv("PINECONE_ENV", "us-east-1")
USDA_API_KEY    = os.getenv("USDA_API_KEY", "DEMO_KEY")

if not GROQ_API_KEY:
    safe_print("[WARNING] GROQ_API_KEY not set. LLM calls will fail.")

# ============================================================
# SECTION 2: INGREDIENT DATASET (Curated + USDA enrichment)
# ============================================================

# ── 2A. Curated base dataset (always present, nutritionally accurate) ──
CURATED_INGREDIENTS: List[Dict[str, Any]] = [
    # ── Protein Sources ────────────────────────────────────
    {"name": "Pea Protein Isolate",    "category": "Protein Source",  "protein": 80.0, "fat":  4.0, "carbs":  5.0, "calories": 380.0, "functional_role": "Protein Source",  "allergen": "None"},
    {"name": "Soy Protein Isolate",    "category": "Protein Source",  "protein": 90.0, "fat":  1.0, "carbs":  1.0, "calories": 370.0, "functional_role": "Protein Source",  "allergen": "Soy"},
    {"name": "Whey Protein Isolate",   "category": "Protein Source",  "protein": 90.0, "fat":  0.5, "carbs":  1.0, "calories": 360.0, "functional_role": "Protein Source",  "allergen": "Milk"},
    {"name": "Brown Rice Protein",     "category": "Protein Source",  "protein": 75.0, "fat":  3.0, "carbs": 10.0, "calories": 370.0, "functional_role": "Protein Source",  "allergen": "None"},
    {"name": "Skim Milk Powder",       "category": "Protein Source",  "protein": 35.0, "fat":  1.0, "carbs": 52.0, "calories": 360.0, "functional_role": "Protein Source",  "allergen": "Milk"},
    {"name": "Egg White Powder",       "category": "Protein Source",  "protein": 80.0, "fat":  0.0, "carbs":  4.0, "calories": 340.0, "functional_role": "Protein Source",  "allergen": "Egg"},
    {"name": "Hemp Protein Powder",    "category": "Protein Source",  "protein": 50.0, "fat": 11.0, "carbs": 18.0, "calories": 370.0, "functional_role": "Protein Source",  "allergen": "None"},
    {"name": "Sunflower Protein",      "category": "Protein Source",  "protein": 70.0, "fat":  8.0, "carbs":  8.0, "calories": 390.0, "functional_role": "Protein Source",  "allergen": "None"},

    # ── Bulking Agents / Carb Bases ────────────────────────
    {"name": "Oat Flour",              "category": "Bulking Agent",   "protein": 12.0, "fat":  7.0, "carbs": 66.0, "calories": 389.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Wheat Flour",            "category": "Bulking Agent",   "protein": 10.0, "fat":  1.0, "carbs": 76.0, "calories": 364.0, "functional_role": "Bulking Agent",   "allergen": "Gluten"},
    {"name": "Maltodextrin",           "category": "Bulking Agent",   "protein":  0.0, "fat":  0.0, "carbs": 95.0, "calories": 380.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Polydextrose",           "category": "Bulking Agent",   "protein":  0.0, "fat":  0.0, "carbs": 90.0, "calories": 100.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Pea Starch",             "category": "Bulking Agent",   "protein":  1.0, "fat":  0.1, "carbs": 88.0, "calories": 350.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Rice Flour",             "category": "Bulking Agent",   "protein":  6.0, "fat":  1.4, "carbs": 80.0, "calories": 360.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Corn Starch",            "category": "Bulking Agent",   "protein":  0.3, "fat":  0.1, "carbs": 91.0, "calories": 381.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Almond Flour",           "category": "Bulking Agent",   "protein": 21.0, "fat": 50.0, "carbs": 22.0, "calories": 590.0, "functional_role": "Bulking Agent",   "allergen": "Tree Nuts"},
    {"name": "Chickpea Flour",         "category": "Bulking Agent",   "protein": 22.0, "fat":  6.0, "carbs": 58.0, "calories": 387.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Tapioca Starch",         "category": "Bulking Agent",   "protein":  0.2, "fat":  0.0, "carbs": 88.0, "calories": 360.0, "functional_role": "Bulking Agent",   "allergen": "None"},

    # ── Lipid Sources ──────────────────────────────────────
    {"name": "Coconut Oil",            "category": "Lipid Source",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 862.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Sunflower Oil",          "category": "Lipid Source",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 884.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "Canola Oil",             "category": "Lipid Source",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 884.0, "functional_role": "Bulking Agent",   "allergen": "None"},
    {"name": "MCT Oil",                "category": "Lipid Source",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 862.0, "functional_role": "Bulking Agent",   "allergen": "None"},

    # ── Emulsifiers ────────────────────────────────────────
    {"name": "Soy Lecithin",           "category": "Emulsifier",      "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 763.0, "functional_role": "Emulsifier",      "allergen": "Soy"},
    {"name": "Sunflower Lecithin",     "category": "Emulsifier",      "protein":  0.0, "fat": 90.0, "carbs":  5.0, "calories": 720.0, "functional_role": "Emulsifier",      "allergen": "None"},
    {"name": "Mono- and Diglycerides", "category": "Emulsifier",      "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 800.0, "functional_role": "Emulsifier",      "allergen": "None"},
    {"name": "Polysorbate 80",         "category": "Emulsifier",      "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Emulsifier",      "allergen": "None"},
    {"name": "DATEM",                  "category": "Emulsifier",      "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 800.0, "functional_role": "Emulsifier",      "allergen": "None"},

    # ── Stabilizers ────────────────────────────────────────
    {"name": "Xanthan Gum",            "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 80.0, "calories": 320.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Guar Gum",               "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 85.0, "calories": 340.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Locust Bean Gum",        "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 80.0, "calories": 320.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Carrageenan",            "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 80.0, "calories": 300.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Gellan Gum",             "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 85.0, "calories": 340.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Methylcellulose",        "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 90.0, "calories": 360.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Pectin",                 "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 90.0, "calories": 360.0, "functional_role": "Stabilizer",      "allergen": "None"},
    {"name": "Sodium Alginate",        "category": "Stabilizer",      "protein":  0.0, "fat":  0.0, "carbs": 80.0, "calories": 320.0, "functional_role": "Stabilizer",      "allergen": "None"},

    # ── Sweeteners ─────────────────────────────────────────
    {"name": "Cane Sugar",             "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 387.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Stevia Extract",         "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Sucralose",              "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Erythritol",             "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories":  20.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Monk Fruit Extract",     "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Allulose",               "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories":  10.0, "functional_role": "Sweetener",       "allergen": "None"},
    {"name": "Coconut Sugar",          "category": "Sweetener",       "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 375.0, "functional_role": "Sweetener",       "allergen": "None"},

    # ── Flavor Systems ─────────────────────────────────────
    {"name": "Cocoa Powder",           "category": "Flavor System",   "protein": 20.0, "fat": 14.0, "carbs": 58.0, "calories": 228.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Vanilla Extract",        "category": "Flavor System",   "protein":  0.0, "fat":  0.0, "carbs": 12.0, "calories": 288.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Natural Chocolate Flavor","category": "Flavor System",  "protein":  0.0, "fat":  0.0, "carbs":  5.0, "calories":  20.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Natural Strawberry Flavor","category": "Flavor System", "protein":  0.0, "fat":  0.0, "carbs":  5.0, "calories":  20.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Natural Caramel Flavor", "category": "Flavor System",   "protein":  0.0, "fat":  0.0, "carbs":  5.0, "calories":  20.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Salt",                   "category": "Flavor System",   "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Flavor System",   "allergen": "None"},
    {"name": "Cinnamon Powder",        "category": "Flavor System",   "protein":  4.0, "fat":  1.2, "carbs": 81.0, "calories": 247.0, "functional_role": "Flavor System",   "allergen": "None"},

    # ── Acidity Controls ───────────────────────────────────
    {"name": "Citric Acid",            "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 300.0, "functional_role": "Acidity Control", "allergen": "None"},
    {"name": "Sodium Bicarbonate",     "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Acidity Control", "allergen": "None"},
    {"name": "Malic Acid",             "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 300.0, "functional_role": "Acidity Control", "allergen": "None"},
    {"name": "Lactic Acid",            "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 300.0, "functional_role": "Acidity Control", "allergen": "None"},
    {"name": "Sodium Citrate",         "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Acidity Control", "allergen": "None"},
    {"name": "Tartaric Acid",          "category": "Acidity Control", "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 300.0, "functional_role": "Acidity Control", "allergen": "None"},

    # ── Preservation ───────────────────────────────────────
    {"name": "Potassium Sorbate",      "category": "Preservation",    "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Sodium Benzoate",        "category": "Preservation",    "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Rosemary Extract",       "category": "Preservation",    "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Mixed Tocopherols",      "category": "Preservation",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 900.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Ascorbic Acid",          "category": "Preservation",    "protein":  0.0, "fat":  0.0, "carbs":100.0, "calories": 400.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Vitamin D3",             "category": "Preservation",    "protein":  0.0, "fat":100.0, "carbs":  0.0, "calories": 900.0, "functional_role": "Preservation",    "allergen": "None"},
    {"name": "Natamycin",              "category": "Preservation",    "protein":  0.0, "fat":  0.0, "carbs":  0.0, "calories":   0.0, "functional_role": "Preservation",    "allergen": "None"},
]

# All 8 functional roles that every formulation must cover
ALL_FUNCTIONAL_ROLES = [
    "Protein Source",
    "Bulking Agent",
    "Emulsifier",
    "Stabilizer",
    "Sweetener",
    "Flavor System",
    "Acidity Control",
    "Preservation",
]

# ── 2B. USDA FDC enrichment (optional, tries to add real USDA items) ──
USDA_NUTRIENT_IDS = {
    "protein":   1003,   # Protein
    "fat":       1004,   # Total lipid (fat)
    "carbs":     1005,   # Carbohydrate, by difference
    "calories":  1008,   # Energy (kcal)
}

def classify_usda_ingredient(name: str) -> Tuple[str, str, str]:
    """Map a raw USDA food description to (category, functional_role, allergen)."""
    n = name.lower()
    if any(x in n for x in ["protein isolate", "protein concentrate", "whey", "casein", "egg white powder"]):
        allergen = "Soy" if "soy" in n else ("Milk" if ("milk" in n or "whey" in n or "casein" in n) else ("Egg" if "egg" in n else "None"))
        return "Protein Source", "Protein Source", allergen
    if any(x in n for x in ["lecithin", "monoglyceride", "diglyceride", "datem", "polysorbate"]):
        return "Emulsifier", "Emulsifier", "Soy" if "soy" in n else "None"
    if any(x in n for x in ["xanthan", "guar gum", "locust bean", "carrageenan", "gellan", "pectin", "alginate", "methylcellulose"]):
        return "Stabilizer", "Stabilizer", "None"
    if any(x in n for x in ["sucralose", "stevia", "erythritol", "aspartame", "saccharin", "monk fruit", "allulose"]):
        return "Sweetener", "Sweetener", "None"
    if any(x in n for x in ["sugar", "dextrose", "fructose", "glucose syrup", "corn syrup"]):
        return "Sweetener", "Sweetener", "None"
    if any(x in n for x in ["cocoa", "vanilla", "flavoring", "flavor", "extract", "spice", "salt", "cinnamon"]):
        return "Flavor System", "Flavor System", "None"
    if any(x in n for x in ["citric acid", "malic acid", "lactic acid", "tartaric acid", "bicarbonate", "citrate", "phosphate"]):
        return "Acidity Control", "Acidity Control", "None"
    if any(x in n for x in ["sorbate", "benzoate", "tocopherol", "ascorbic", "natamycin", "rosemary extract"]):
        return "Preservation", "Preservation", "None"
    # Default: Bulking Agent
    allergen = "Gluten" if ("wheat" in n or "flour" in n) else ("Milk" if ("milk" in n or "dairy" in n) else ("Soy" if "soy" in n else ("Tree Nuts" if any(x in n for x in ["almond", "cashew", "hazelnut", "walnut", "peanut"]) else "None")))
    return "Bulking Agent", "Bulking Agent", allergen


def fetch_usda_enrichment(api_key: str) -> List[Dict[str, Any]]:
    """
    Query USDA FoodData Central for food science relevant ingredients.
    Uses nutrient IDs (not string names) for reliable macro extraction.
    Returns a list of dicts in the same schema as CURATED_INGREDIENTS.
    """
    search_queries = [
        "whey protein isolate",
        "soy protein concentrate",
        "pea protein",
        "xanthan gum",
        "guar gum",
        "soy lecithin",
        "maltodextrin",
        "erythritol",
        "sucralose",
        "cocoa powder unsweetened",
        "citric acid",
        "ascorbic acid",
        "potassium sorbate",
        "tapioca starch",
    ]

    enriched = []
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    for query in tqdm(search_queries, desc="USDA enrichment"):
        try:
            params = {
                "query": query,
                "api_key": api_key,
                "pageSize": 3,
                "dataType": "Branded,SR Legacy,Foundation",  # all types
            }
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            foods = resp.json().get("foods", [])

            for food in foods:
                name = food.get("description", "").strip()
                if not name:
                    continue

                # Extract nutrients using nutrient IDs
                nutrients_raw = food.get("foodNutrients", [])
                nmap: Dict[int, float] = {}
                for n in nutrients_raw:
                    nid   = n.get("nutrientId") or n.get("nutrientNumber")
                    value = n.get("value", 0.0)
                    try:
                        nmap[int(nid)] = float(value)
                    except (TypeError, ValueError):
                        pass

                protein  = nmap.get(USDA_NUTRIENT_IDS["protein"],  0.0)
                fat      = nmap.get(USDA_NUTRIENT_IDS["fat"],       0.0)
                carbs    = nmap.get(USDA_NUTRIENT_IDS["carbs"],     0.0)
                calories = nmap.get(USDA_NUTRIENT_IDS["calories"],  0.0)

                # Skip if all macros are zero (data is useless)
                if protein == 0.0 and fat == 0.0 and carbs == 0.0 and calories == 0.0:
                    continue

                category, role, allergen = classify_usda_ingredient(name)
                enriched.append({
                    "name":            name,
                    "category":        category,
                    "protein":         round(protein,  2),
                    "fat":             round(fat,      2),
                    "carbs":           round(carbs,    2),
                    "calories":        round(calories, 2),
                    "functional_role": role,
                    "allergen":        allergen,
                })

            time.sleep(0.3)   # Be polite to USDA API

        except Exception as e:
            safe_print(f"  [USDA] Query '{query}' failed: {e}")

    return enriched


def prepare_dataset() -> pd.DataFrame:
    """
    Build the ingredient DataFrame:
      1. Start with curated base (always reliable).
      2. Try to enrich with USDA API items that have real macro values.
      3. Deduplicate by name.
      4. Save to CSV.
    """
    safe_print("\n[DATASET] Building ingredient database...")
    df = pd.DataFrame(CURATED_INGREDIENTS)
    safe_print(f"  Curated base: {len(df)} ingredients loaded.")

    if USDA_API_KEY and USDA_API_KEY != "DEMO_KEY":
        safe_print("  Attempting USDA FDC enrichment (real API key detected)...")
        try:
            usda_items = fetch_usda_enrichment(USDA_API_KEY)
            if usda_items:
                usda_df = pd.DataFrame(usda_items)
                existing_lower = set(df["name"].str.lower())
                new_items = usda_df[~usda_df["name"].str.lower().isin(existing_lower)]
                df = pd.concat([df, new_items], ignore_index=True)
                safe_print(f"  USDA enrichment added {len(new_items)} new ingredients.")
            else:
                safe_print("  USDA returned no usable items; using curated data only.")
        except Exception as e:
            safe_print(f"  USDA enrichment failed: {e}. Using curated data only.")
    else:
        safe_print("  No real USDA_API_KEY set; skipping USDA enrichment.")

    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    csv_path = os.path.join(script_dir, "food_ingredients_clean.csv")
    df.to_csv(csv_path, index=False)
    safe_print(f"  Final dataset: {len(df)} unique ingredients → saved to {csv_path}\n")
    return df


# ============================================================
# SECTION 3: DOCUMENT BUILDING
# ============================================================
def build_documents(df: pd.DataFrame) -> List[LCDocument]:
    """
    Convert each ingredient row into a LangChain Document.
    page_content is enriched with semantic context so vector
    similarity search works on intent-level queries.
    """
    docs = []
    for _, row in df.iterrows():
        # Rich semantic text — helps retrieval match queries like
        # "chocolate beverage emulsifier" to "Sunflower Lecithin"
        page_content = (
            f"Ingredient Name: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"Functional Role: {row['functional_role']}\n"
            f"Allergen: {row['allergen']}\n"
            f"Protein: {row['protein']} g per 100g\n"
            f"Fat: {row['fat']} g per 100g\n"
            f"Carbohydrates: {row['carbs']} g per 100g\n"
            f"Calories: {row['calories']} kcal per 100g\n"
            f"Use cases: This ingredient is commonly used as a {row['functional_role'].lower()} "
            f"in food product formulations including beverages, bars, bakery, dairy alternatives, "
            f"confections, and sports nutrition products."
        )
        metadata = {
            "name":            str(row["name"]),
            "category":        str(row["category"]),
            "functional_role": str(row["functional_role"]),
            "allergen":        str(row["allergen"]),
            "protein":         float(row["protein"]),
            "fat":             float(row["fat"]),
            "carbs":           float(row["carbs"]),
            "calories":        float(row["calories"]),
        }
        docs.append(LCDocument(page_content=page_content, metadata=metadata))
    return docs


# ============================================================
# SECTION 4: IN-MEMORY VECTOR STORE (Pinecone fallback)
# ============================================================
class InMemoryVectorStore:
    """
    Cosine-similarity vector store backed by sentence-transformers.
    Supports per-category filtered search — the key fix for RAG quality.
    """
    def __init__(self, documents: List[LCDocument], embeddings: HuggingFaceEmbeddings):
        self.documents  = documents
        self.embeddings = embeddings
        safe_print("  Pre-computing embeddings for in-memory vector store...")
        texts = [d.page_content for d in documents]
        self.doc_embeddings = np.array(self.embeddings.embed_documents(texts))
        safe_print(f"  Done. {len(documents)} embeddings stored (dim={self.doc_embeddings.shape[1]}).")

    def _cosine_search(self, query: str, k: int,
                       role_filter: Optional[str] = None) -> List[LCDocument]:
        q_emb = np.array(self.embeddings.embed_query(query))

        # Apply optional role filter
        if role_filter:
            indices  = [i for i, d in enumerate(self.documents)
                        if d.metadata.get("functional_role") == role_filter]
            if not indices:
                indices = list(range(len(self.documents)))
        else:
            indices = list(range(len(self.documents)))

        sub_embs = self.doc_embeddings[indices]
        dots     = sub_embs @ q_emb
        norms    = np.linalg.norm(sub_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
        sims     = dots / norms
        top_k    = np.argsort(sims)[::-1][:k]
        return [self.documents[indices[i]] for i in top_k]

    def category_aware_search(self, query: str, k_per_role: int = 2) -> List[LCDocument]:
        """
        THE CORE FIX: retrieve top-k per functional role so that
        every role is guaranteed to be represented in the context.
        """
        results: List[LCDocument] = []
        seen_names: set = set()
        for role in ALL_FUNCTIONAL_ROLES:
            role_docs = self._cosine_search(query, k=k_per_role, role_filter=role)
            for doc in role_docs:
                name = doc.metadata.get("name", "")
                if name not in seen_names:
                    results.append(doc)
                    seen_names.add(name)
        return results

    # Compatibility shim: as_retriever used by simple invocations
    def similarity_search(self, query: str, k: int = 8) -> List[LCDocument]:
        return self._cosine_search(query, k)

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        store = self
        k = (search_kwargs or {}).get("k", 8)
        class _Retriever:
            def invoke(self, query: str) -> List[LCDocument]:
                return store.similarity_search(query, k)
        return _Retriever()


# ============================================================
# SECTION 5: PINECONE SETUP (with in-memory fallback)
# ============================================================
def setup_vector_store(df: pd.DataFrame,
                       embeddings: HuggingFaceEmbeddings) -> InMemoryVectorStore:
    """
    Attempts Pinecone setup; falls back to InMemoryVectorStore.
    Returns the vector store object (always InMemoryVectorStore shape
    so the rest of the code is identical either way).
    """
    documents = build_documents(df)
    index_name = "tcs-food-formulator"

    if PINECONE_API_KEY and PINECONE_API_KEY not in ("", "your_pinecone_api_key_here"):
        safe_print("[PINECONE] Real API key detected; attempting Pinecone setup...")
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore

            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing:
                safe_print(f"  Creating index '{index_name}'...")
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
                )
                time.sleep(5)  # wait for index to be ready

            pc_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY,
            )

            # Upsert documents in batches
            batch_size = 50
            ids = [d.metadata["name"].lower().replace(" ", "_") for d in documents]
            for i in tqdm(range(0, len(documents), batch_size), desc="Pinecone upsert"):
                pc_store.add_documents(documents[i:i+batch_size], ids=ids[i:i+batch_size])

            safe_print(f"  Pinecone ready: {len(documents)} vectors upserted.\n")

            # Wrap Pinecone store to expose category_aware_search
            # by building an InMemoryVectorStore on top (embeddings already computed)
            safe_print("  Building local category-aware index on top of Pinecone data...")
            mem_store = InMemoryVectorStore(documents, embeddings)
            return mem_store   # category_aware_search is the critical path

        except Exception as e:
            safe_print(f"  Pinecone setup failed: {e}. Falling back to in-memory store.\n")

    safe_print("[VECTOR STORE] Using in-memory cosine similarity store.")
    return InMemoryVectorStore(documents, embeddings)


# ============================================================
# SECTION 6: LANGGRAPH AGENTIC RAG PIPELINE
# ============================================================

# ── Agent State ────────────────────────────────────────────
class AgentState(TypedDict):
    query:         str
    context_docs:  List[LCDocument]
    raw_response:  str

# ── Global references (set in main) ────────────────────────
vector_store_ref: Optional[InMemoryVectorStore] = None
llm_ref:          Optional[ChatGroq] = None
fallback_llm_ref: Optional[ChatGroq] = None


# ── Node 1: Category-Aware Retrieval ──────────────────────
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    FIXED retrieval: queries each functional role separately so that
    EVERY role (Protein Source, Emulsifier, Stabilizer, etc.) is
    guaranteed to appear in the context passed to the LLM.
    Previously this was a single cosine search that accidentally
    omitted functional ingredients because their text similarity
    to a product query (e.g. "chocolate beverage") was low.
    """
    query = state["query"]
    safe_print(f"\n  [RETRIEVE] Category-aware multi-role retrieval for: '{query}'")
    docs = vector_store_ref.category_aware_search(query, k_per_role=2)
    safe_print(f"  [RETRIEVE] Retrieved {len(docs)} ingredients covering all roles.")
    for doc in docs:
        role = doc.metadata.get("functional_role", "?")
        name = doc.metadata.get("name", "?")
        safe_print(f"    [{role}] {name}")
    return {"context_docs": docs}


# ── Node 2: LLM Generation ────────────────────────────────
SYSTEM_PROMPT = """\
You are a Senior Food Scientist at TCS Research designing formulation reports for clients.
Generate a recipe formulation report in the EXACT plain-text format below.
Rules:
  1. All percentages must sum to exactly 100.00%.
  2. Use ONLY ingredients from the provided context list.
  3. Allergen filtering: if the query says vegan, exclude Milk/Egg. If gluten-free, exclude Gluten.
  4. Calculate Protein (g), Sugar (g), and Calories (kcal) from the formulation percentages and
     the per-100g macros supplied in the ingredient context.
     Formula: macro_contribution = (ingredient_concentration_pct / 100) * ingredient_macro_per_100g
     Sum contributions from all ingredients for total macros.
  5. Output raw plain text ONLY — no markdown, no asterisks, no backticks.

Format Template:
====================================================================
TCS FOOD FORMULATOR – GENERATED FORMULATION REPORT
Constraint-Driven Multi-Objective Recipe Generation
====================================================================

Recipe ID      : <e.g. CHOC_BEV_001>
Recipe Name    : <Descriptive Name>
Category       : <Product Category>
Version        : V1.0

====================================================================
1. COMPLETE FORMULATION COMPOSITION
====================================================================

BULK INGREDIENTS
--------------------------------------------------------------------
Ingredient                     Category           Conc. (%)

<Ingredient 1>                 <Category>         <XX.XX>
<Ingredient 2>                 <Category>         <XX.XX>

Subtotal (Bulk)                                   <XX.XX> %

--------------------------------------------------------------------

FUNCTIONAL INGREDIENTS
--------------------------------------------------------------------
Ingredient                     Functional Role    Conc. (%)

<Ingredient 1>                 <Role>             <XX.XX>
<Ingredient 2>                 <Role>             <XX.XX>

Subtotal (Functional)                              <XX.XX> %

====================================================================
TOTAL FORMULATION = 100.00 %
====================================================================

====================================================================
2. FUNCTIONAL ROLE COVERAGE
====================================================================

Role                     Ingredient               Coverage

Protein Source           <Ingredient>              [YES/NO]
Bulking Agent            <Ingredient>              [YES/NO]
Emulsifier               <Ingredient>              [YES/NO]
Stabilizer               <Ingredient>              [YES/NO]
Sweetener                <Ingredient>              [YES/NO]
Flavor System            <Ingredient>              [YES/NO]
Acidity Control          <Ingredient>              [YES/NO]
Preservation             <Ingredient>              [YES/NO]

Functional Completeness Score (FCS): <XX.X> % (<X>/8 roles covered)
Assessment: <FUNCTIONALLY COMPLETE (>=90%) / NEEDS REVIEW (<90%)>

====================================================================
3. PRODUCT PERFORMANCE PREDICTION
====================================================================

Protein                 : <XX.X> g / 100g product
Total Fat               : <XX.X> g / 100g product
Carbohydrates           : <XX.X> g / 100g product
Calories                : <XXX> kcal / 100g product

Texture                 : <Description>
Mouthfeel               : <Description>
Viscosity               : <Description>
Shelf-life              : ~<X> months

====================================================================
4. PROCESS RECOMMENDATION
====================================================================

STEP 1: <Description>
STEP 2: <Description>
STEP 3: <Description>
STEP 4: Homogenization: <RPM> rpm x <duration> min
STEP 5: Heat treatment: <temp>C x <duration> sec

====================================================================
FINAL STATUS
====================================================================

Recipe Status: READY FOR VIRTUAL VALIDATION
====================================================================
"""


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the formulation report. The context_docs now contain
    one ingredient per functional role (guaranteed by retrieve_node),
    so the LLM has everything it needs to fill all roles.
    """
    query = state["query"]
    docs  = state["context_docs"]

    # Build structured context block
    context_lines = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        context_lines.append(
            f"[{i}] Name: {m['name']} | Role: {m['functional_role']} | "
            f"Protein: {m['protein']}g | Fat: {m['fat']}g | "
            f"Carbs: {m['carbs']}g | Calories: {m['calories']} kcal | "
            f"Allergen: {m['allergen']}"
        )
    context_str = "\n".join(context_lines)

    user_msg = (
        f"Product Request: {query}\n\n"
        f"Available Ingredients (use ONLY these):\n{context_str}\n\n"
        f"Generate the formulation report now."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    safe_print("  [GENERATE] Invoking LLM...")
    try:
        response = llm_ref.invoke(messages)
    except Exception as e:
        safe_print(f"  [GENERATE] Primary LLM failed: {e}. Using fallback LLM...")
        response = fallback_llm_ref.invoke(messages)

    return {"raw_response": response.content}


# ── Build LangGraph ────────────────────────────────────────
def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ============================================================
# SECTION 7: QUERY ENGINE
# ============================================================
def run_query(app: Any, query: str) -> str:
    """Execute a single formulation query through the LangGraph pipeline."""
    safe_print(f"\n{'='*68}")
    safe_print(f"QUERY: {query}")
    safe_print(f"{'='*68}")
    try:
        result = app.invoke({"query": query})
        return result["raw_response"]
    except Exception as e:
        safe_print(f"  [ERROR] Agent workflow failed: {e}")
        safe_print("  [FALLBACK] Running direct LLM call...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Product Request: {query}"),
        ]
        try:
            return llm_ref.invoke(messages).content
        except Exception as ex:
            return fallback_llm_ref.invoke(messages).content


# ============================================================
# SECTION 8: MAIN
# ============================================================
def main():
    global vector_store_ref, llm_ref, fallback_llm_ref

    safe_print("\n" + "="*68)
    safe_print("  TCS FOOD FORMULATOR — CORRECTED RAG PIPELINE")
    safe_print("="*68 + "\n")

    # ── Step 1: Dataset ──────────────────────────────────────
    df = prepare_dataset()

    # ── Step 2: Embeddings ───────────────────────────────────
    safe_print("[EMBEDDINGS] Loading sentence-transformers/all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    safe_print("  Embeddings model loaded.\n")

    # ── Step 3: Vector Store ─────────────────────────────────
    vector_store_ref = setup_vector_store(df, embeddings)

    # ── Step 4: LLMs ────────────────────────────────────────
    safe_print("[LLM] Initializing Groq LLMs...")
    try:
        llm_ref = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
        )
        safe_print("  Primary LLM: llama-3.3-70b-versatile")
    except Exception as e:
        safe_print(f"  Primary LLM init failed ({e}); falling back...")
        llm_ref = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
        )

    fallback_llm_ref = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
    )
    safe_print("  Fallback LLM: llama-3.1-8b-instant\n")

    # ── Step 5: Build Graph ──────────────────────────────────
    safe_print("[GRAPH] Compiling LangGraph pipeline...")
    app = build_graph()
    safe_print("  LangGraph compiled: retrieve → generate → END\n")

    # ── Step 6: Run Queries ──────────────────────────────────
    queries = [
        "Generate a high protein low sugar chocolate beverage formulation",
        "Suggest ingredients for a vegan energy bar with 20g protein",
        "What emulsifiers are suitable for a plant-based milk alternative?",
        "Design a gluten-free bakery formulation for sandwich bread",
    ]

    reports: List[str] = []
    for idx, q in enumerate(queries, 1):
        report = run_query(app, q)
        reports.append(report)
        safe_print(f"\n[REPORT {idx}]\n{report}")
        safe_print("\n" + "="*68)

    # ── Step 7: Summary ──────────────────────────────────────
    safe_print("\n" + "="*68)
    safe_print("  TCS FOOD FORMULATOR — RUN SUMMARY")
    safe_print("="*68)
    safe_print(f"  Ingredients in database : {len(df)}")
    safe_print(f"  Functional roles covered: {len(ALL_FUNCTIONAL_ROLES)}")
    safe_print(f"  Queries processed       : {len(queries)}")
    safe_print(f"  CSV saved               : food_ingredients_clean.csv")
    safe_print(f"  Retrieval strategy      : Category-aware (2 docs/role x 8 roles = 16 docs)")
    safe_print("="*68 + "\n")


if __name__ == "__main__":
    main()