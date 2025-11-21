## California Procurement Dataset Schema

This document summarizes the structure of the California Department of General Services purchase order extract that powers the procurement assistant. The raw CSV contains 919,734 rows covering fiscal years 2012–2015 (156 MB, 31 columns). Each row represents a purchase-order line with UNSPSC classifications, supplier metadata, amounts, and dates.

### Key Entities
- **Purchase Order** – identified by `purchase_order_number`
- **Department** – state agency placing the order (`department_name`)
- **Supplier** – vendor fulfilling the order (`supplier_name`, `supplier_code`)
- **Commodity Classification** – UNSPSC hierarchy (segment → family → class → commodity)
- **Financials** – unit/total price, quantities, fiscal years, acquisition methods

### Column Reference

| Original Column | Normalized Field | Type | Description / Usage | Notes |
| --- | --- | --- | --- | --- |
| Creation Date | `creation_date` | datetime | Date the PO was created. | Parsed from `MM/DD/YYYY`; 100% populated. |
| Purchase Date | `purchase_date` | datetime | Actual purchase date (can lag creation). | ~4.7% missing. |
| Fiscal Year | `fiscal_year` | string | Fiscal period label (e.g., `2014-2015`). | Treat as categorical dimension. |
| LPA Number | `lpa_number` | string | Leveraged Procurement Agreement identifier. | 74% missing; mostly blank for non-LPA orders. |
| Purchase Order Number | `purchase_order_number` | string | Unique PO identifier per line. | Repeats when a PO has multiple line items. |
| Requisition Number | `requisition_number` | string | Internal requisition reference. | 96% missing. |
| Acquisition Type | `acquisition_type` | string | High-level category (IT Goods, Non-IT Services, etc.). | 5 categories; good for filters. |
| Sub-Acquisition Type | `sub_acquisition_type` | string | Finer acquisition classification. | 80% missing. |
| Acquisition Method | `acquisition_method` | string | Procurement vehicle (Statewide Contract, Informal Competitive, etc.). | Rich categorical field. |
| Sub-Acquisition Method | `sub_acquisition_method` | string | Additional method qualifier. | 91% missing. |
| Department Name | `department_name` | string | Ordering state department. | 101 unique agencies. |
| Supplier Code | `supplier_code` | string | DGS supplier identifier. | Mostly populated; treat as string to preserve leading zeros. |
| Supplier Name | `supplier_name` | string | Vendor name. | 4,840 unique suppliers. |
| Supplier Qualifications | `supplier_qualifications` | string | Certification stack (SB, DVBE, etc.). | 58% missing/blank. |
| Supplier Zip Code | `supplier_zip_code` | string | Supplier ZIP/postal code. | 20% missing. |
| CalCard | `calcard` | categorical (YES/NO) | Indicates CalCard purchase. | YES ≈ 1.4% of rows. |
| Item Name | `item_name` | string | Short descriptor for the line item. | Often generic (e.g., “Contract”). |
| Item Description | `item_description` | string | Free-text detail. | 0.05% missing. |
| Quantity | `quantity` | float | Number of units purchased. | Contains decimals for partial units. |
| Unit Price | `unit_price` | float | Price per unit (USD). | Parsed from currency string. |
| Total Price | `total_price` | float | Extended line value (USD). | Parsed from currency string; can exceed $1 B. |
| Classification Codes | `classification_codes` | string | Raw UNSPSC code path. | Sparse mismatched values (0.25% missing). |
| Normalized UNSPSC | `normalized_unspsc` | string | Cleaned UNSPSC commodity code. | Combine with hierarchy columns. |
| Commodity Title | `commodity_title` | string | Commodity (lowest UNSPSC level) label. | Minor (<1%) missingness. |
| Class | `class` | string | UNSPSC class code. | Minor (<1%) missingness. |
| Class Title | `class_title` | string | Human-readable class label. | Mirrors `class`. |
| Family | `family` | string | UNSPSC family code. | Minor (<1%) missingness. |
| Family Title | `family_title` | string | UNSPSC family label. | Minor (<1%) missingness. |
| Segment | `segment` | string | UNSPSC segment code. | Minor (<1%) missingness. |
| Segment Title | `segment_title` | string | UNSPSC segment label. | Minor (<1%) missingness. |
| Location | `location` | string | Ordering location / region. | ~20% missing. |

### Derived & Helper Fields
- **Snake-case normalization**: All column names are lowercased with underscores for MongoDB compatibility (see `src/data_loader.py`).
- **Currency normalization**: `unit_price` and `total_price` are cleaned to floating-point values by stripping `$`, commas, and other symbols before ingestion.
- **Temporal shards**: Additional features such as `Creation Year`, `Creation Quarter`, etc., can be derived during analysis or query generation.

### Hierarchies & Relationships
- **UNSPSC hierarchy**: `segment` → `family` → `class` → `commodity_title` describe the same code at increasing granularity. Use the shared prefixes of `normalized_unspsc` for drilldowns.
- **Purchase orders and line items**: Multiple records can share a `purchase_order_number`; aggregations should group on this field when computing PO-level metrics.
- **Supplier qualifications**: Values combine multiple certifications (e.g., `CA-MB CA-SB`); treat as multi-valued tokens when filtering.

### Data Quality Notes
- High missingness fields (`requisition_number`, `sub_acquisition_method`, `sub_acquisition_type`) should be optional in prompts and agents.
- Monetary fields have extreme ranges—top 5% of orders account for ~97% of spend. Consider log-scaling or percentile buckets for analytics.
- Duplicated rows are rare (~0.06%); duplicates typically represent identical uploads and can be removed safely.
- Fiscal years are strings (e.g., `2014-2015`). Convert to ordered categories if doing time-series analysis.

### Usage Tips for the AI Assistant
- Prefer numeric filters on `total_price` for spend analytics; `unit_price` is useful for normalization across quantities.
- Date filters should handle both fiscal (`fiscal_year`) and calendar (`creation_date`, `purchase_date`) notions.
- When generating MongoDB pipelines, use `$match` on `acquisition_type`, `department_name`, `segment_title`, etc., plus `$group`/`$sort` for aggregations.
- Implement fuzzy matching for `department_name` and `supplier_name` as values may contain punctuation or inconsistent casing.

This schema should be used in conjunction with the refreshed EDA notebook (`notebooks/eda.ipynb`) and the MongoDB validation script (`scripts/validate_mongodb_schema.py`) to keep the ingestion pipeline reliable.


