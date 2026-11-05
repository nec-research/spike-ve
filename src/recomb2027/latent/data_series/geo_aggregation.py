#   Forecasting SARS-CoV-2 Spike Evolution in a Protein Language Model Latent Space
#
#   File:     geo_aggregation.py
#   Authors:  Raman Siarheyeu (raman.siarheyeu@neclab.eu)
#             Miguel Sánchez-Palomino (miguel.palomino@neclab.eu)
#             Nithya Bhasker (nithya.bhasker@neclab.eu)
#             Catherine Chaput (catherine.chaput@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2026, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on
# the original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted to you under this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
"""Map isolation_country labels to coarser geography buckets (e.g. continent)."""

from __future__ import annotations

from typing import Literal

GeoAggregation = Literal["country", "continent"]

GEO_AGGREGATION_CHOICES: tuple[str, ...] = ("country", "continent")
METADATA_UNK = "<unk>"

GEO_AGGREGATION_CHOICES: tuple[str, ...] = ("country", "continent")

CONTINENT_NAMES: tuple[str, ...] = (
    "Africa",
    "Antarctica",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
)

# Normalised country name (lower case, underscores → spaces) → continent.
# Covers BV-BRC / GISAID-style names seen in df_labels.csv.
_COUNTRY_TO_CONTINENT: dict[str, str] = {
    # Africa
    "algeria": "Africa",
    "angola": "Africa",
    "benin": "Africa",
    "botswana": "Africa",
    "burkina faso": "Africa",
    "burundi": "Africa",
    "cameroon": "Africa",
    "cape verde": "Africa",
    "central african republic": "Africa",
    "chad": "Africa",
    "comoros": "Africa",
    "congo": "Africa",
    "cote d'ivoire": "Africa",
    "democratic republic of the congo": "Africa",
    "djibouti": "Africa",
    "egypt": "Africa",
    "equatorial guinea": "Africa",
    "eritrea": "Africa",
    "eswatini": "Africa",
    "ethiopia": "Africa",
    "gabon": "Africa",
    "gambia": "Africa",
    "ghana": "Africa",
    "guinea": "Africa",
    "guinea bissau": "Africa",
    "kenya": "Africa",
    "lesotho": "Africa",
    "liberia": "Africa",
    "libya": "Africa",
    "madagascar": "Africa",
    "malawi": "Africa",
    "mali": "Africa",
    "mauritania": "Africa",
    "mauritius": "Africa",
    "morocco": "Africa",
    "mozambique": "Africa",
    "namibia": "Africa",
    "niger": "Africa",
    "nigeria": "Africa",
    "republic of the congo": "Africa",
    "rwanda": "Africa",
    "senegal": "Africa",
    "seychelles": "Africa",
    "sierra leone": "Africa",
    "somalia": "Africa",
    "south africa": "Africa",
    "south sudan": "Africa",
    "sudan": "Africa",
    "tanzania": "Africa",
    "togo": "Africa",
    "tunisia": "Africa",
    "uganda": "Africa",
    "zambia": "Africa",
    "zimbabwe": "Africa",
    # Asia
    "afghanistan": "Asia",
    "armenia": "Asia",
    "azerbaijan": "Asia",
    "bahrain": "Asia",
    "bangladesh": "Asia",
    "bhutan": "Asia",
    "brunei": "Asia",
    "cambodia": "Asia",
    "china": "Asia",
    "cyprus": "Asia",
    "georgia": "Asia",
    "hong kong": "Asia",
    "india": "Asia",
    "indonesia": "Asia",
    "iran": "Asia",
    "iraq": "Asia",
    "israel": "Asia",
    "japan": "Asia",
    "jordan": "Asia",
    "kazakhstan": "Asia",
    "kuwait": "Asia",
    "kyrgyzstan": "Asia",
    "laos": "Asia",
    "lebanon": "Asia",
    "macau": "Asia",
    "malaysia": "Asia",
    "maldives": "Asia",
    "mongolia": "Asia",
    "myanmar": "Asia",
    "nepal": "Asia",
    "north korea": "Asia",
    "oman": "Asia",
    "pakistan": "Asia",
    "palestine": "Asia",
    "philippines": "Asia",
    "qatar": "Asia",
    "saudi arabia": "Asia",
    "singapore": "Asia",
    "south korea": "Asia",
    "sri lanka": "Asia",
    "syria": "Asia",
    "taiwan": "Asia",
    "tajikistan": "Asia",
    "thailand": "Asia",
    "timor leste": "Asia",
    "turkey": "Asia",
    "turkmenistan": "Asia",
    "united arab emirates": "Asia",
    "uzbekistan": "Asia",
    "viet nam": "Asia",
    "vietnam": "Asia",
    "yemen": "Asia",
    # Europe
    "albania": "Europe",
    "andorra": "Europe",
    "austria": "Europe",
    "belarus": "Europe",
    "belgium": "Europe",
    "bosnia and herzegovina": "Europe",
    "bulgaria": "Europe",
    "croatia": "Europe",
    "czech republic": "Europe",
    "czechia": "Europe",
    "denmark": "Europe",
    "estonia": "Europe",
    "finland": "Europe",
    "france": "Europe",
    "germany": "Europe",
    "greece": "Europe",
    "hungary": "Europe",
    "iceland": "Europe",
    "ireland": "Europe",
    "italy": "Europe",
    "kosovo": "Europe",
    "latvia": "Europe",
    "liechtenstein": "Europe",
    "lithuania": "Europe",
    "luxembourg": "Europe",
    "malta": "Europe",
    "moldova": "Europe",
    "monaco": "Europe",
    "montenegro": "Europe",
    "netherlands": "Europe",
    "north macedonia": "Europe",
    "norway": "Europe",
    "poland": "Europe",
    "portugal": "Europe",
    "romania": "Europe",
    "russia": "Europe",
    "san marino": "Europe",
    "serbia": "Europe",
    "slovakia": "Europe",
    "slovenia": "Europe",
    "spain": "Europe",
    "sweden": "Europe",
    "switzerland": "Europe",
    "ukraine": "Europe",
    "united kingdom": "Europe",
    "vatican city": "Europe",
    # North America
    "antigua and barbuda": "North America",
    "bahamas": "North America",
    "barbados": "North America",
    "belize": "North America",
    "canada": "North America",
    "costa rica": "North America",
    "cuba": "North America",
    "dominica": "North America",
    "dominican republic": "North America",
    "el salvador": "North America",
    "grenada": "North America",
    "guatemala": "North America",
    "haiti": "North America",
    "honduras": "North America",
    "jamaica": "North America",
    "mexico": "North America",
    "nicaragua": "North America",
    "panama": "North America",
    "puerto rico": "North America",
    "saint kitts and nevis": "North America",
    "saint lucia": "North America",
    "saint vincent and the grenadines": "North America",
    "trinidad and tobago": "North America",
    "usa": "North America",
    "united states": "North America",
    "united states of america": "North America",
    # Oceania
    "australia": "Oceania",
    "fiji": "Oceania",
    "kiribati": "Oceania",
    "marshall islands": "Oceania",
    "micronesia": "Oceania",
    "nauru": "Oceania",
    "new caledonia": "Oceania",
    "new zealand": "Oceania",
    "palau": "Oceania",
    "papua new guinea": "Oceania",
    "samoa": "Oceania",
    "solomon islands": "Oceania",
    "tonga": "Oceania",
    "tuvalu": "Oceania",
    "vanuatu": "Oceania",
    # South America
    "argentina": "South America",
    "bolivia": "South America",
    "brazil": "South America",
    "chile": "South America",
    "colombia": "South America",
    "ecuador": "South America",
    "guyana": "South America",
    "paraguay": "South America",
    "peru": "South America",
    "suriname": "South America",
    "uruguay": "South America",
    "venezuela": "South America",
    # Antarctica (rare in public health data)
    "antarctica": "Antarctica",
}

# Common aliases not covered by underscore normalisation alone.
_COUNTRY_ALIASES: dict[str, str] = {
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "u.s.a.": "usa",
    "u.s.": "usa",
    "us": "usa",
    "uae": "united arab emirates",
    "drc": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",
    "cote divoire": "cote d'ivoire",
    "ivory coast": "cote d'ivoire",
    "korea, south": "south korea",
    "korea, north": "north korea",
    "republic of korea": "south korea",
    "democratic people's republic of korea": "north korea",
}


def normalize_country_key(raw: object) -> str:
    """Canonical lower-case country key for lookup (``South_Africa`` → ``south africa``)."""
    s = str(raw).strip().replace("_", " ")
    if not s or s.lower() in {"nan", "none", ""}:
        return METADATA_UNK.lower()
    key = " ".join(s.lower().split())
    return _COUNTRY_ALIASES.get(key, key)


def country_to_continent(country: str) -> str:
    """Map a raw country label to a continent name, or ``<unk>`` if unknown."""
    key = normalize_country_key(country)
    if key == METADATA_UNK.lower():
        return METADATA_UNK
    continent = _COUNTRY_TO_CONTINENT.get(key)
    if continent is None:
        return METADATA_UNK
    return continent


def normalize_geo_label(
    raw: object,
    *,
    geo_aggregation: GeoAggregation = "country",
) -> str:
    """Return the geography label used for diffusion vocab / conditioning."""
    if geo_aggregation == "country":
        s = str(raw).strip().replace("_", " ")
        if not s or s.lower() in {"nan", "none", ""}:
            return METADATA_UNK
        return s
    if geo_aggregation == "continent":
        return country_to_continent(str(raw))
    raise ValueError(
        f"Unknown geo_aggregation={geo_aggregation!r}; "
        f"expected one of {GEO_AGGREGATION_CHOICES}"
    )
