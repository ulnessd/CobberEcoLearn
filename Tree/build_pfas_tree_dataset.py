#!/usr/bin/env python3
"""
build_pfas_tree_dataset.py

Build a small PFAS teaching dataset for an ecology decision-tree chapter.

The script:
  1. starts from a curated list of PFAS names/abbreviations,
  2. queries PubChem PUG-REST,
  3. retrieves molecular properties including XLogP,
  4. computes simple teaching descriptors from formula/name/SMILES,
  5. writes CSV files for CobberEcoTree-style activities.

Output files:
  pfas_tree_dataset.csv              all resolved records
  pfas_tree_dataset_for_tree.csv     records with non-missing XLogP
  pfas_manual_subset.csv             small manual-sorting subset
  pfas_fetch_report.csv              resolved/skipped/error report

Dependencies:
  pip install pandas requests

Run:
  python build_pfas_tree_dataset.py

Optional:
  python build_pfas_tree_dataset.py --outdir PFASData --sleep 0.25
  python build_pfas_tree_dataset.py --limit 50

Notes:
  - XLogP is a computed PubChem property. It is useful for teaching partitioning
    behavior, but it is not a complete PFAS environmental-fate descriptor.
  - PFAS names can be ambiguous. The script stores the resolved PubChem CID
    and records any failed queries in the report file.
  - The simple descriptors are for teaching decision trees, not regulatory use.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPERTY_LIST = [
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
]


@dataclass
class PFASQuery:
    abbreviation: str
    preferred_name: str
    query_names: List[str]
    pfas_class_hint: str = ""


# The list intentionally mixes legacy PFAS, short-chain PFAS, sulfonates,
# fluorotelomer compounds, ether PFAS, and several precursors.
#
# Not every name is guaranteed to resolve in PubChem. That is okay: the report
# file records failures, and the tree dataset uses the successful records.
PFAS_QUERIES: List[PFASQuery] = [
    # Perfluoroalkyl carboxylic acids (PFCAs)
    PFASQuery("TFA", "Trifluoroacetic acid", ["trifluoroacetic acid", "TFA"], "PFCA"),
    PFASQuery("PFPrA", "Perfluoropropionic acid", ["perfluoropropionic acid", "pentafluoropropionic acid", "PFPrA"], "PFCA"),
    PFASQuery("PFBA", "Perfluorobutanoic acid", ["perfluorobutanoic acid", "heptafluorobutyric acid", "PFBA"], "PFCA"),
    PFASQuery("PFPeA", "Perfluoropentanoic acid", ["perfluoropentanoic acid", "nonafluoropentanoic acid", "PFPeA"], "PFCA"),
    PFASQuery("PFHxA", "Perfluorohexanoic acid", ["perfluorohexanoic acid", "undecafluorohexanoic acid", "PFHxA"], "PFCA"),
    PFASQuery("PFHpA", "Perfluoroheptanoic acid", ["perfluoroheptanoic acid", "tridecafluoroheptanoic acid", "PFHpA"], "PFCA"),
    PFASQuery("PFOA", "Perfluorooctanoic acid", ["perfluorooctanoic acid", "pentadecafluorooctanoic acid", "PFOA"], "PFCA"),
    PFASQuery("PFNA", "Perfluorononanoic acid", ["perfluorononanoic acid", "heptadecafluorononanoic acid", "PFNA"], "PFCA"),
    PFASQuery("PFDA", "Perfluorodecanoic acid", ["perfluorodecanoic acid", "nonadecafluorodecanoic acid", "PFDA"], "PFCA"),
    PFASQuery("PFUnDA", "Perfluoroundecanoic acid", ["perfluoroundecanoic acid", "henicosafluoroundecanoic acid", "PFUnDA"], "PFCA"),
    PFASQuery("PFDoDA", "Perfluorododecanoic acid", ["perfluorododecanoic acid", "tricosafluorododecanoic acid", "PFDoDA"], "PFCA"),
    PFASQuery("PFTrDA", "Perfluorotridecanoic acid", ["perfluorotridecanoic acid", "pentacosafluorotridecanoic acid", "PFTrDA"], "PFCA"),
    PFASQuery("PFTeDA", "Perfluorotetradecanoic acid", ["perfluorotetradecanoic acid", "heptacosafluorotetradecanoic acid", "PFTeDA"], "PFCA"),
    PFASQuery("PFHxDA", "Perfluorohexadecanoic acid", ["perfluorohexadecanoic acid", "PFHxDA"], "PFCA"),
    PFASQuery("PFOcDA", "Perfluorooctadecanoic acid", ["perfluorooctadecanoic acid", "PFOcDA"], "PFCA"),

    # Perfluoroalkyl sulfonic acids (PFSAs)
    PFASQuery("PFMS", "Perfluoromethanesulfonic acid", ["perfluoromethanesulfonic acid", "trifluoromethanesulfonic acid", "triflic acid"], "PFSA"),
    PFASQuery("PFEtS", "Perfluoroethanesulfonic acid", ["perfluoroethanesulfonic acid", "pentafluoroethanesulfonic acid", "PFEtS"], "PFSA"),
    PFASQuery("PFPrS", "Perfluoropropanesulfonic acid", ["perfluoropropanesulfonic acid", "heptafluoropropanesulfonic acid", "PFPrS"], "PFSA"),
    PFASQuery("PFBS", "Perfluorobutanesulfonic acid", ["perfluorobutanesulfonic acid", "nonafluorobutanesulfonic acid", "PFBS"], "PFSA"),
    PFASQuery("PFPeS", "Perfluoropentanesulfonic acid", ["perfluoropentanesulfonic acid", "undecafluoropentanesulfonic acid", "PFPeS"], "PFSA"),
    PFASQuery("PFHxS", "Perfluorohexanesulfonic acid", ["perfluorohexanesulfonic acid", "tridecafluorohexanesulfonic acid", "PFHxS"], "PFSA"),
    PFASQuery("PFHpS", "Perfluoroheptanesulfonic acid", ["perfluoroheptanesulfonic acid", "pentadecafluoroheptanesulfonic acid", "PFHpS"], "PFSA"),
    PFASQuery("PFOS", "Perfluorooctanesulfonic acid", ["perfluorooctanesulfonic acid", "heptadecafluorooctanesulfonic acid", "PFOS"], "PFSA"),
    PFASQuery("PFNS", "Perfluorononanesulfonic acid", ["perfluorononanesulfonic acid", "nonadecafluorononanesulfonic acid", "PFNS"], "PFSA"),
    PFASQuery("PFDS", "Perfluorodecanesulfonic acid", ["perfluorodecanesulfonic acid", "henicosafluorodecanesulfonic acid", "PFDS"], "PFSA"),
    PFASQuery("PFDoS", "Perfluorododecanesulfonic acid", ["perfluorododecanesulfonic acid", "PFDoS"], "PFSA"),

    # Fluorotelomer sulfonic acids
    PFASQuery("4:2 FTS", "4:2 Fluorotelomer sulfonic acid", ["4:2 fluorotelomer sulfonic acid", "4:2 FTS"], "FTS"),
    PFASQuery("6:2 FTS", "6:2 Fluorotelomer sulfonic acid", ["6:2 fluorotelomer sulfonic acid", "6:2 FTS"], "FTS"),
    PFASQuery("8:2 FTS", "8:2 Fluorotelomer sulfonic acid", ["8:2 fluorotelomer sulfonic acid", "8:2 FTS"], "FTS"),
    PFASQuery("10:2 FTS", "10:2 Fluorotelomer sulfonic acid", ["10:2 fluorotelomer sulfonic acid", "10:2 FTS"], "FTS"),

    # Fluorotelomer alcohols
    PFASQuery("4:2 FTOH", "4:2 Fluorotelomer alcohol", ["4:2 fluorotelomer alcohol", "4:2 FTOH"], "FTOH"),
    PFASQuery("6:2 FTOH", "6:2 Fluorotelomer alcohol", ["6:2 fluorotelomer alcohol", "6:2 FTOH"], "FTOH"),
    PFASQuery("8:2 FTOH", "8:2 Fluorotelomer alcohol", ["8:2 fluorotelomer alcohol", "8:2 FTOH"], "FTOH"),
    PFASQuery("10:2 FTOH", "10:2 Fluorotelomer alcohol", ["10:2 fluorotelomer alcohol", "10:2 FTOH"], "FTOH"),

    # Fluorotelomer carboxylic acids / unsaturated acids
    PFASQuery("3:3 FTCA", "3:3 Fluorotelomer carboxylic acid", ["3:3 fluorotelomer carboxylic acid", "3:3 FTCA"], "FTCA"),
    PFASQuery("5:3 FTCA", "5:3 Fluorotelomer carboxylic acid", ["5:3 fluorotelomer carboxylic acid", "5:3 FTCA"], "FTCA"),
    PFASQuery("7:3 FTCA", "7:3 Fluorotelomer carboxylic acid", ["7:3 fluorotelomer carboxylic acid", "7:3 FTCA"], "FTCA"),
    PFASQuery("9:3 FTCA", "9:3 Fluorotelomer carboxylic acid", ["9:3 fluorotelomer carboxylic acid", "9:3 FTCA"], "FTCA"),
    PFASQuery("6:2 FTUCA", "6:2 Fluorotelomer unsaturated carboxylic acid", ["6:2 fluorotelomer unsaturated carboxylic acid", "6:2 FTUCA"], "FTUCA"),
    PFASQuery("8:2 FTUCA", "8:2 Fluorotelomer unsaturated carboxylic acid", ["8:2 fluorotelomer unsaturated carboxylic acid", "8:2 FTUCA"], "FTUCA"),

    # Perfluoroalkyl sulfonamides and sulfonamido ethanols
    PFASQuery("FOSA", "Perfluorooctanesulfonamide", ["perfluorooctanesulfonamide", "PFOSA", "FOSA"], "FOSA"),
    PFASQuery("N-MeFOSA", "N-Methyl perfluorooctanesulfonamide", ["N-methyl perfluorooctanesulfonamide", "N-MeFOSA"], "FOSA"),
    PFASQuery("N-EtFOSA", "N-Ethyl perfluorooctanesulfonamide", ["N-ethyl perfluorooctanesulfonamide", "N-EtFOSA"], "FOSA"),
    PFASQuery("FOSE", "Perfluorooctanesulfonamidoethanol", ["perfluorooctanesulfonamidoethanol", "FOSE"], "FOSE"),
    PFASQuery("N-MeFOSE", "N-Methyl perfluorooctanesulfonamidoethanol", ["N-methyl perfluorooctanesulfonamidoethanol", "N-MeFOSE"], "FOSE"),
    PFASQuery("N-EtFOSE", "N-Ethyl perfluorooctanesulfonamidoethanol", ["N-ethyl perfluorooctanesulfonamidoethanol", "N-EtFOSE"], "FOSE"),

    # Ether PFAS / replacement PFAS
    PFASQuery("HFPO-DA", "Hexafluoropropylene oxide dimer acid", ["hexafluoropropylene oxide dimer acid", "HFPO-DA", "GenX"], "Ether acid"),
    PFASQuery("ADONA", "ADONA", ["ADONA", "ammonium 4,8-dioxa-3H-perfluorononanoate", "4,8-dioxa-3H-perfluorononanoic acid"], "Ether acid"),
    PFASQuery("PFMOAA", "Perfluoromethoxyacetic acid", ["perfluoromethoxyacetic acid", "PFMOAA"], "Ether acid"),
    PFASQuery("PFECA B", "Perfluoro-2-methoxyacetic acid", ["perfluoro-2-methoxyacetic acid"], "Ether acid"),
    PFASQuery("PFO2HxA", "Perfluoro-3,5-dioxahexanoic acid", ["perfluoro-3,5-dioxahexanoic acid", "PFO2HxA"], "Ether acid"),
    PFASQuery("PFO3OA", "Perfluoro-3,5,7-trioxaoctanoic acid", ["perfluoro-3,5,7-trioxaoctanoic acid", "PFO3OA"], "Ether acid"),
    PFASQuery("Nafion BP2", "Nafion byproduct 2", ["Nafion byproduct 2"], "Ether sulfonate"),
    PFASQuery("11Cl-PF3OUdS", "11-chloroeicosafluoro-3-oxaundecane-1-sulfonic acid", ["11-chloroeicosafluoro-3-oxaundecane-1-sulfonic acid", "11Cl-PF3OUdS"], "Ether sulfonate"),
    PFASQuery("9Cl-PF3ONS", "9-chlorohexadecafluoro-3-oxanonane-1-sulfonic acid", ["9-chlorohexadecafluoro-3-oxanonane-1-sulfonic acid", "9Cl-PF3ONS"], "Ether sulfonate"),

    # Perfluoroethers / neutral fluorinated ethers
    PFASQuery("PFPE-1", "Perfluoropolyether alcohol", ["perfluoropolyether alcohol"], "PFPE"),
    PFASQuery("Perfluorobutyl methyl ether", "Perfluorobutyl methyl ether", ["perfluorobutyl methyl ether"], "Ether"),
    PFASQuery("Perfluorohexyl ethyl ether", "Perfluorohexyl ethyl ether", ["perfluorohexyl ethyl ether"], "Ether"),

    # Additional perfluoroalkyl substances / related molecules
    PFASQuery("PFHxI", "Perfluorohexyl iodide", ["perfluorohexyl iodide", "1-iodoperfluorohexane"], "Perfluoroalkyl iodide"),
    PFASQuery("PFOI", "Perfluorooctyl iodide", ["perfluorooctyl iodide", "1-iodoperfluorooctane"], "Perfluoroalkyl iodide"),
    PFASQuery("PFDAI", "Perfluorodecyl iodide", ["perfluorodecyl iodide", "1-iodoperfluorodecane"], "Perfluoroalkyl iodide"),
    PFASQuery("PFHxBr", "Perfluorohexyl bromide", ["perfluorohexyl bromide", "1-bromoperfluorohexane"], "Perfluoroalkyl bromide"),
    PFASQuery("PFOBr", "Perfluorooctyl bromide", ["perfluorooctyl bromide", "1-bromoperfluorooctane"], "Perfluoroalkyl bromide"),
    PFASQuery("Perfluorobutane", "Perfluorobutane", ["perfluorobutane"], "Perfluoroalkane"),
    PFASQuery("Perfluorohexane", "Perfluorohexane", ["perfluorohexane"], "Perfluoroalkane"),
    PFASQuery("Perfluorooctane", "Perfluorooctane", ["perfluorooctane"], "Perfluoroalkane"),
    PFASQuery("Perfluorodecalin", "Perfluorodecalin", ["perfluorodecalin"], "Perfluoroalkane"),
    PFASQuery("Perfluorotributylamine", "Perfluorotributylamine", ["perfluorotributylamine"], "Perfluoroamine"),
    PFASQuery("Perfluorotripropylamine", "Perfluorotripropylamine", ["perfluorotripropylamine"], "Perfluoroamine"),

    # More telomer-related precursors
    PFASQuery("6:2 FTA", "6:2 Fluorotelomer acrylate", ["6:2 fluorotelomer acrylate", "6:2 FTAcr"], "Acrylate"),
    PFASQuery("8:2 FTA", "8:2 Fluorotelomer acrylate", ["8:2 fluorotelomer acrylate", "8:2 FTAcr"], "Acrylate"),
    PFASQuery("6:2 FTMA", "6:2 Fluorotelomer methacrylate", ["6:2 fluorotelomer methacrylate", "6:2 FTMA"], "Methacrylate"),
    PFASQuery("8:2 FTMA", "8:2 Fluorotelomer methacrylate", ["8:2 fluorotelomer methacrylate", "8:2 FTMA"], "Methacrylate"),
    PFASQuery("6:2 diPAP", "6:2 Fluorotelomer phosphate diester", ["6:2 fluorotelomer phosphate diester", "6:2 diPAP"], "Phosphate ester"),
    PFASQuery("8:2 diPAP", "8:2 Fluorotelomer phosphate diester", ["8:2 fluorotelomer phosphate diester", "8:2 diPAP"], "Phosphate ester"),
    PFASQuery("6:2/8:2 diPAP", "6:2/8:2 Fluorotelomer phosphate diester", ["6:2/8:2 fluorotelomer phosphate diester", "6:2/8:2 diPAP"], "Phosphate ester"),

    # Additional acids often seen in PFAS lists
    PFASQuery("PFMBA", "Perfluoro-4-methoxybutanoic acid", ["perfluoro-4-methoxybutanoic acid", "PFMBA"], "Ether acid"),
    PFASQuery("PFMPA", "Perfluoro-3-methoxypropanoic acid", ["perfluoro-3-methoxypropanoic acid", "PFMPA"], "Ether acid"),
    PFASQuery("PFEEA", "Perfluoroethoxyethanoic acid", ["perfluoroethoxyethanoic acid", "PFEEA"], "Ether acid"),
    PFASQuery("PFPrOPrA", "Perfluoropropoxypropanoic acid", ["perfluoropropoxypropanoic acid", "PFPrOPrA"], "Ether acid"),
    PFASQuery("PFO4DA", "Perfluoro-3,5,7,9-tetraoxadecanoic acid", ["perfluoro-3,5,7,9-tetraoxadecanoic acid", "PFO4DA"], "Ether acid"),

    # Chlorinated polyfluoroether sulfonic acids / related
    PFASQuery("6:2 Cl-PFESA", "6:2 chlorinated polyfluoroether sulfonic acid", ["6:2 chlorinated polyfluoroether sulfonic acid", "6:2 Cl-PFESA"], "Ether sulfonate"),
    PFASQuery("8:2 Cl-PFESA", "8:2 chlorinated polyfluoroether sulfonic acid", ["8:2 chlorinated polyfluoroether sulfonic acid", "8:2 Cl-PFESA"], "Ether sulfonate"),
]


def query_pubchem_cids(name: str, timeout: float = 20.0) -> List[int]:
    encoded = quote(name)
    url = f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data = response.json()
    return [int(cid) for cid in data.get("IdentifierList", {}).get("CID", [])]


def fetch_pubchem_properties(cid: int, timeout: float = 20.0) -> Dict[str, object]:
    props = ",".join(PROPERTY_LIST)
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    records = data.get("PropertyTable", {}).get("Properties", [])
    if not records:
        raise ValueError(f"No property record returned for CID {cid}")
    return records[0]


def parse_formula_counts(formula: str) -> Dict[str, int]:
    """
    Parse a simple molecular formula such as C8HF15O2.
    Handles element symbols followed by optional integer counts.
    Does not handle nested parentheses; PubChem molecular formulae rarely need
    that for this teaching workflow.
    """
    counts: Dict[str, int] = {}
    if not isinstance(formula, str):
        return counts
    for element, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        count = int(count_str) if count_str else 1
        counts[element] = counts.get(element, 0) + count
    return counts


def contains_any(text: str, needles: Iterable[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def infer_pfas_class(q: PFASQuery, formula: str, smiles: str) -> str:
    if q.pfas_class_hint:
        return q.pfas_class_hint

    text = f"{q.abbreviation} {q.preferred_name} {smiles}".lower()
    if "sulfon" in text:
        return "Sulfonate/sulfonamide"
    if "carbox" in text or "anoic acid" in text or "acetic acid" in text:
        return "Carboxylic acid"
    if "fluorotelomer" in text:
        return "Fluorotelomer"
    if "ether" in text or "oxa" in text:
        return "Ether PFAS"
    return "Other PFAS"


def compute_teaching_descriptors(q: PFASQuery, props: Dict[str, object]) -> Dict[str, object]:
    formula = str(props.get("MolecularFormula", ""))
    smiles = str(props.get("CanonicalSMILES", ""))
    counts = parse_formula_counts(formula)

    c = counts.get("C", 0)
    f = counts.get("F", 0)
    o = counts.get("O", 0)
    s = counts.get("S", 0)
    n = counts.get("N", 0)
    cl = counts.get("Cl", 0)
    br = counts.get("Br", 0)
    i_count = counts.get("I", 0)
    p = counts.get("P", 0)

    text = f"{q.abbreviation} {q.preferred_name} {q.pfas_class_hint} {smiles}".lower()

    carboxylate_flag = int(
        "carbox" in text
        or "anoic acid" in text
        or "acetic acid" in text
        or "C(=O)O" in smiles
        or "C(=O)[O-]" in smiles
    )
    sulfonate_flag = int("sulfonic acid" in text or "sulfonate" in text or "S(=O)(=O)O" in smiles or "S(=O)(=O)[O-]" in smiles)
    sulfonamide_flag = int("sulfonamide" in text or "sulfonamido" in text)
    ether_flag = int("ether" in text or "oxa" in text or "O" in smiles.replace("=O", ""))

    # Very rough teaching estimate:
    # oxygen atoms not explained by the common functional groups.
    functional_o = 0
    if carboxylate_flag:
        functional_o += 2
    if sulfonate_flag:
        functional_o += 3
    if sulfonamide_flag:
        functional_o += 2
    if "alcohol" in text or "ethanol" in text:
        functional_o += 1
    if p > 0 or "phosphate" in text:
        functional_o += 4
    ether_oxygen_count = max(0, o - functional_o)

    # Approximate "fluorinated carbons" for a simple teaching feature. For many
    # PFAS, F count is more robust than trying to infer exact fluorinated carbons
    # without RDKit, so keep both and let the tree choose.
    estimated_fluorinated_carbons = min(c, max(0, round((f + 1) / 2)))

    return {
        "PFAS_Class": infer_pfas_class(q, formula, smiles),
        "Carbon_Count": c,
        "Fluorine_Count": f,
        "Oxygen_Count": o,
        "Sulfur_Count": s,
        "Nitrogen_Count": n,
        "Chlorine_Count": cl,
        "Bromine_Count": br,
        "Iodine_Count": i_count,
        "Phosphorus_Count": p,
        "Estimated_Fluorinated_Carbons": estimated_fluorinated_carbons,
        "Ether_Oxygen_Count": ether_oxygen_count,
        "Carboxylate_Flag": carboxylate_flag,
        "Sulfonate_Flag": sulfonate_flag,
        "Sulfonamide_Flag": sulfonamide_flag,
        "Ether_Flag": ether_flag,
        "Halogen_Count": f + cl + br + i_count,
    }


def resolve_one(q: PFASQuery, sleep_s: float, timeout: float) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    report = {
        "Abbreviation": q.abbreviation,
        "Preferred_Name": q.preferred_name,
        "Status": "not tried",
        "Resolved_Query": "",
        "CID": "",
        "Message": "",
    }

    for name in q.query_names:
        try:
            cids = query_pubchem_cids(name, timeout=timeout)
            time.sleep(sleep_s)
            if not cids:
                continue

            cid = cids[0]
            props = fetch_pubchem_properties(cid, timeout=timeout)
            time.sleep(sleep_s)

            descriptors = compute_teaching_descriptors(q, props)
            row: Dict[str, object] = {
                "Abbreviation": q.abbreviation,
                "Preferred_Name": q.preferred_name,
                "Resolved_Query": name,
                "PubChem_CID": cid,
                "MolecularFormula": props.get("MolecularFormula"),
                "CanonicalSMILES": props.get("CanonicalSMILES"),
                "IsomericSMILES": props.get("IsomericSMILES"),
                "MolecularWeight": props.get("MolecularWeight"),
                "XLogP": props.get("XLogP"),
                "TPSA": props.get("TPSA"),
                "HBondDonorCount": props.get("HBondDonorCount"),
                "HBondAcceptorCount": props.get("HBondAcceptorCount"),
                "RotatableBondCount": props.get("RotatableBondCount"),
                "HeavyAtomCount": props.get("HeavyAtomCount"),
            }
            row.update(descriptors)

            report.update({
                "Status": "resolved",
                "Resolved_Query": name,
                "CID": cid,
                "Message": "ok",
            })
            return row, report

        except requests.RequestException as exc:
            report.update({
                "Status": "error",
                "Resolved_Query": name,
                "Message": f"request error: {exc}",
            })
            # Continue trying alternate names if available.
            time.sleep(sleep_s)

        except Exception as exc:
            report.update({
                "Status": "error",
                "Resolved_Query": name,
                "Message": f"error: {exc}",
            })
            time.sleep(sleep_s)

    if report["Status"] == "not tried":
        report["Status"] = "unresolved"
        report["Message"] = "no query names tried"
    elif report["Status"] != "resolved":
        report["Status"] = "unresolved"
        if not report["Message"]:
            report["Message"] = "no PubChem CID found for supplied names"
    return None, report


def choose_manual_subset(df_tree: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """
    Pick a small, diverse manual-sorting deck.

    Prefer well-known compounds if available, then fill in by spreading across
    the XLogP range and PFAS classes.
    """
    preferred = [
        "PFBA", "PFHxA", "PFOA", "PFNA", "PFDA",
        "PFBS", "PFHxS", "PFOS",
        "4:2 FTS", "6:2 FTS", "8:2 FTS",
        "4:2 FTOH", "6:2 FTOH", "8:2 FTOH",
        "HFPO-DA", "ADONA", "FOSA",
    ]

    chosen_indices: List[int] = []
    for abbr in preferred:
        hits = df_tree.index[df_tree["Abbreviation"].astype(str) == abbr].tolist()
        if hits and hits[0] not in chosen_indices:
            chosen_indices.append(hits[0])
        if len(chosen_indices) >= n:
            break

    # Fill by XLogP quantiles if needed.
    if len(chosen_indices) < n and not df_tree.empty:
        sorted_df = df_tree.sort_values("XLogP").copy()
        if len(sorted_df) > 1:
            positions = [round(i) for i in pd.Series(range(len(sorted_df))).quantile([j / max(1, n - 1) for j in range(n)]).tolist()]
        else:
            positions = [0]
        for pos in positions:
            idx = sorted_df.index[int(max(0, min(pos, len(sorted_df) - 1)))]
            if idx not in chosen_indices:
                chosen_indices.append(idx)
            if len(chosen_indices) >= n:
                break

    return df_tree.loc[chosen_indices].copy().reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PFAS decision-tree teaching dataset from PubChem.")
    parser.add_argument("--outdir", default=".", help="Output directory for CSV files.")
    parser.add_argument("--sleep", type=float, default=0.20, help="Seconds to sleep between PubChem requests.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PFAS queries for testing. 0 means no limit.")
    parser.add_argument("--manual-n", type=int, default=12, help="Number of records for the manual sorting subset.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    queries = PFAS_QUERIES[: args.limit] if args.limit and args.limit > 0 else PFAS_QUERIES

    rows: List[Dict[str, object]] = []
    reports: List[Dict[str, object]] = []
    seen_cids = set()

    print(f"Querying PubChem for {len(queries)} PFAS entries...")
    for idx, q in enumerate(queries, start=1):
        print(f"[{idx:03d}/{len(queries):03d}] {q.abbreviation}: {q.preferred_name}")
        row, report = resolve_one(q, sleep_s=args.sleep, timeout=args.timeout)
        reports.append(report)
        if row is not None:
            cid = row.get("PubChem_CID")
            if cid in seen_cids:
                report["Status"] = "duplicate"
                report["Message"] = f"duplicate CID {cid}; not added to dataset"
            else:
                seen_cids.add(cid)
                rows.append(row)

    df = pd.DataFrame(rows)
    report_df = pd.DataFrame(reports)

    if not df.empty:
        # Numeric cleanup.
        numeric_cols = [
            "MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount",
            "RotatableBondCount", "HeavyAtomCount", "Carbon_Count", "Fluorine_Count",
            "Oxygen_Count", "Sulfur_Count", "Nitrogen_Count", "Chlorine_Count",
            "Bromine_Count", "Iodine_Count", "Phosphorus_Count", "Estimated_Fluorinated_Carbons",
            "Ether_Oxygen_Count", "Carboxylate_Flag", "Sulfonate_Flag",
            "Sulfonamide_Flag", "Ether_Flag", "Halogen_Count",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convenient sorting for human inspection.
        sort_cols = [c for c in ["PFAS_Class", "Carbon_Count", "Fluorine_Count", "Abbreviation"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    dataset_path = outdir / "pfas_tree_dataset.csv"
    report_path = outdir / "pfas_fetch_report.csv"
    tree_path = outdir / "pfas_tree_dataset_for_tree.csv"
    manual_path = outdir / "pfas_manual_subset.csv"

    df.to_csv(dataset_path, index=False)
    report_df.to_csv(report_path, index=False)

    if not df.empty and "XLogP" in df.columns:
        df_tree = df.dropna(subset=["XLogP"]).copy().reset_index(drop=True)
    else:
        df_tree = pd.DataFrame()

    df_tree.to_csv(tree_path, index=False)

    if not df_tree.empty:
        manual_df = choose_manual_subset(df_tree, n=args.manual_n)
    else:
        manual_df = pd.DataFrame()
    manual_df.to_csv(manual_path, index=False)

    print("\nDone.")
    print(f"Resolved records: {len(df)}")
    print(f"Records with XLogP: {len(df_tree)}")
    print(f"Manual subset records: {len(manual_df)}")
    print(f"Wrote: {dataset_path}")
    print(f"Wrote: {tree_path}")
    print(f"Wrote: {manual_path}")
    print(f"Wrote: {report_path}")

    if len(df_tree) < 40:
        print("\nWARNING: Fewer than 40 records with XLogP were produced.")
        print("Check pfas_fetch_report.csv for unresolved names or try adding more query names.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
