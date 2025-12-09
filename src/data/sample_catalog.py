"""
TD SYNNEX Partner Product Catalog - 5K Cisco/HP/Dell EU Dataset
Complete with pricing, specs, regions, revenue trends, and partner segments
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Product categories and specifications
VENDORS = {
    "Cisco": {
        "categories": ["Switches", "Routers", "Wireless", "Security", "Collaboration"],
        "models": {
            "Switches": ["Catalyst-9200", "Catalyst-9300", "Catalyst-9400", "Catalyst-9500", "Catalyst-9600", 
                        "Nexus-9300", "Nexus-9500", "Meraki-MS120", "Meraki-MS250", "Meraki-MS350"],
            "Routers": ["ISR-1000", "ISR-4000", "ASR-1000", "ASR-9000", "Catalyst-8200", "Catalyst-8300"],
            "Wireless": ["Meraki-MR36", "Meraki-MR46", "Meraki-MR56", "Catalyst-9100", "Aironet-2800"],
            "Security": ["Firepower-1000", "Firepower-2100", "ASA-5500", "Umbrella-SIG", "SecureX"],
            "Collaboration": ["Webex-Room-Kit", "Webex-Board", "Webex-Desk", "IP-Phone-8800", "Headset-700"]
        }
    },
    "HP": {
        "categories": ["Servers", "Storage", "Networking", "Workstations", "Accessories"],
        "models": {
            "Servers": ["ProLiant-DL380", "ProLiant-DL360", "ProLiant-ML350", "Synergy-480", "Edgeline-EL8000"],
            "Storage": ["Nimble-HF20", "Nimble-AF40", "Primera-A630", "StoreEasy-1660", "MSA-2062"],
            "Networking": ["Aruba-2930F", "Aruba-6300", "Aruba-CX-8360", "FlexNetwork-5130", "OfficeConnect"],
            "Workstations": ["Z4-G4", "Z6-G4", "Z8-G4", "ZBook-Fury", "ZBook-Studio"],
            "Accessories": ["Smart-Tank-750", "LaserJet-Pro", "EliteDisplay-E27", "Poly-Studio", "Thunderbolt-Dock"]
        }
    },
    "Dell": {
        "categories": ["Servers", "Storage", "Networking", "Laptops", "Desktops"],
        "models": {
            "Servers": ["PowerEdge-R750", "PowerEdge-R650", "PowerEdge-R550", "PowerEdge-T550", "PowerEdge-MX7000"],
            "Storage": ["PowerStore-500", "PowerStore-1000", "PowerScale-F200", "Unity-XT-380", "ECS-EX500"],
            "Networking": ["PowerSwitch-S5248", "PowerSwitch-Z9432", "PowerSwitch-N3248", "SmartFabric-Director"],
            "Laptops": ["Latitude-7440", "Latitude-5540", "Precision-7680", "XPS-15", "Inspiron-16"],
            "Desktops": ["OptiPlex-7010", "OptiPlex-5000", "Precision-3660", "Precision-5860", "XPS-Desktop"]
        }
    }
}

EU_REGIONS = ["CZ", "DE", "PL", "AT", "SK", "HU", "NL", "BE", "FR", "IT", "ES", "UK", "SE", "DK", "NO", "FI"]
PARTNER_SEGMENTS = ["Enterprise", "SMB", "SOHO", "Government", "Education", "Healthcare"]
STOCK_STATUS = ["In Stock", "Low Stock", "Pre-Order", "Out of Stock"]

def generate_product_catalog(n_products: int = 5000) -> pd.DataFrame:
    """Generate a realistic TD SYNNEX product catalog with 5k products"""
    
    products = []
    product_id = 1000
    
    for _ in range(n_products):
        # Select random vendor and category
        vendor = random.choice(list(VENDORS.keys()))
        category = random.choice(VENDORS[vendor]["categories"])
        model = random.choice(VENDORS[vendor]["models"][category])
        
        # Generate realistic pricing based on category
        base_prices = {
            "Switches": (25000, 450000),
            "Routers": (30000, 500000),
            "Wireless": (8000, 85000),
            "Security": (45000, 650000),
            "Collaboration": (15000, 180000),
            "Servers": (85000, 850000),
            "Storage": (120000, 1200000),
            "Networking": (20000, 280000),
            "Workstations": (55000, 350000),
            "Accessories": (2000, 45000),
            "Laptops": (25000, 120000),
            "Desktops": (18000, 85000)
        }
        
        price_range = base_prices.get(category, (10000, 100000))
        price_czk = random.randint(price_range[0], price_range[1])
        
        # Generate specs
        specs = generate_specs(vendor, category, model)
        
        # Revenue trend (Q3 performance)
        revenue_trend = random.uniform(-15, 35)
        
        # Stock level
        stock_qty = random.choices(
            [random.randint(0, 500), random.randint(1, 50), random.randint(51, 200), 0],
            weights=[0.6, 0.2, 0.15, 0.05]
        )[0]
        
        stock_status = "In Stock" if stock_qty > 50 else "Low Stock" if stock_qty > 0 else "Out of Stock"
        
        products.append({
            "product_id": f"TD-{product_id:06d}",
            "vendor": vendor,
            "category": category,
            "model": model,
            "full_name": f"{vendor} {model}",
            "specs": specs,
            "price_czk": price_czk,
            "price_eur": int(price_czk / 25.5),  # CZK to EUR conversion
            "region": random.choice(EU_REGIONS),
            "partner_segment": random.choice(PARTNER_SEGMENTS),
            "revenue_trend_q3": round(revenue_trend, 1),
            "stock_qty": stock_qty,
            "stock_status": stock_status,
            "warranty_years": random.choice([1, 2, 3, 5]),
            "lead_time_days": random.choice([0, 1, 3, 5, 7, 14, 21, 30]),
            "margin_pct": round(random.uniform(8, 28), 1),
            "certification_required": random.choice([True, False]),
            "eco_rating": random.choice(["A+", "A", "B", "C"]),
            "description": generate_description(vendor, category, model, specs)
        })
        
        product_id += 1
    
    return pd.DataFrame(products)


def generate_specs(vendor: str, category: str, model: str) -> str:
    """Generate realistic product specifications"""
    
    specs_templates = {
        "Switches": [
            f"48-port Gigabit, PoE+, {random.choice([4, 8, 12])}x 10G SFP+, {random.randint(176, 640)}Gbps switching",
            f"{random.choice([24, 48])}-port Multi-Gig, UPOE, Stackable, {random.choice(['Layer 2', 'Layer 3'])} managed",
            f"Access switch, {random.randint(8, 48)} ports, {random.choice(['2.5G', '5G', '10G'])} uplinks, Cloud-managed"
        ],
        "Routers": [
            f"Enterprise router, {random.randint(2, 8)} WAN ports, {random.randint(1, 10)}Gbps throughput, SD-WAN ready",
            f"ISP-grade, {random.randint(100, 500)}Gbps capacity, BGP/MPLS support, Dual PSU"
        ],
        "Servers": [
            f"2U Rack, {random.choice(['Intel Xeon Gold', 'Intel Xeon Platinum', 'AMD EPYC'])}, {random.choice([128, 256, 512, 1024])}GB RAM, {random.randint(2, 8)}x NVMe",
            f"4U Tower, Dual CPU, {random.randint(16, 64)} DIMM slots, {random.randint(8, 24)} drive bays, Redundant PSU"
        ],
        "Storage": [
            f"All-Flash Array, {random.randint(10, 500)}TB raw, {random.randint(100, 500)}K IOPS, NVMe-oF support",
            f"Hybrid storage, {random.randint(50, 1000)}TB, Dedup 5:1, Compression, Replication included"
        ],
        "Wireless": [
            f"WiFi 6E, Tri-band, {random.randint(2, 8)} spatial streams, {random.randint(100, 300)} concurrent clients",
            f"Enterprise AP, Cloud-managed, Built-in analytics, {random.choice(['Indoor', 'Outdoor', 'Industrial'])} rated"
        ],
        "Security": [
            f"NGFW, {random.randint(1, 10)}Gbps threat throughput, IPS/IDS, URL filtering, {random.randint(50, 1000)} IPsec tunnels",
            f"Unified threat management, {random.randint(500, 5000)} users, Sandboxing, Zero-day protection"
        ]
    }
    
    default_specs = f"Enterprise-grade {category.lower()}, {vendor} certified, 24/7 support included"
    templates = specs_templates.get(category, [default_specs])
    return random.choice(templates)


def generate_description(vendor: str, category: str, model: str, specs: str) -> str:
    """Generate product description for RAG context"""
    
    benefits = [
        "industry-leading performance",
        "enterprise-grade reliability",
        "simplified management",
        "reduced TCO",
        "enhanced security",
        "scalable architecture",
        "future-proof investment",
        "seamless integration"
    ]
    
    use_cases = {
        "Enterprise": "large-scale deployments across multiple sites",
        "SMB": "small to medium business environments",
        "Government": "public sector and government agencies",
        "Education": "educational institutions and campuses",
        "Healthcare": "healthcare facilities and medical centers"
    }
    
    segment = random.choice(list(use_cases.keys()))
    selected_benefits = random.sample(benefits, 3)
    
    return (
        f"The {vendor} {model} delivers {selected_benefits[0]} and {selected_benefits[1]} "
        f"for {use_cases[segment]}. Key specifications: {specs}. "
        f"This solution offers {selected_benefits[2]} with {vendor}'s proven technology stack. "
        f"Ideal for partners targeting {segment} customers seeking reliable {category.lower()} solutions."
    )


def get_sample_catalog() -> pd.DataFrame:
    """Get or generate the sample catalog"""
    return generate_product_catalog(5000)


def get_mini_catalog() -> pd.DataFrame:
    """Get a smaller catalog for quick testing"""
    return generate_product_catalog(500)


if __name__ == "__main__":
    # Generate and display sample
    catalog = get_mini_catalog()
    print(f"Generated {len(catalog)} products")
    print(catalog.head(10))
    print(f"\nVendor distribution:\n{catalog['vendor'].value_counts()}")
    print(f"\nCategory distribution:\n{catalog['category'].value_counts()}")
