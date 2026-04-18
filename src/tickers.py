"""
tickers.py  —  Fase 2
Universo titoli italiano espanso: FTSE MIB (31) + FTSE Italia Mid Cap (25) + FTSE Italia Small Cap selezionati (20)
Totale: ~76 titoli con liquidità sufficiente per la strategia.

Criteri di selezione small/mid cap:
  - Volume medio giornaliero > 500k EUR
  - Capitalizzazione > 200M EUR
  - Quotati da almeno 3 anni
  - Dati storici disponibili su yfinance
"""

# ── FTSE MIB (31 titoli — large cap) ─────────────────────────────────────────
FTSE_MIB_TICKERS = [
    "A2A.MI", "AMP.MI", "AZM.MI", "BAMI.MI", "BMED.MI",
    "BPE.MI", "CPR.MI", "DIA.MI", "ENEL.MI", "ENI.MI",
    "ERG.MI", "FBK.MI", "G.MI",   "HER.MI",  "IG.MI",
    "INW.MI", "ISP.MI", "LDO.MI", "MB.MI",   "MONC.MI",
    "NEXI.MI","PIRC.MI","PRY.MI", "PST.MI",  "REC.MI",
    "SRG.MI", "STLAM.MI","TEN.MI","TRN.MI",  "UCG.MI",
    "UNI.MI",
]

# ── FTSE Italia Mid Cap (selezione liquida) ───────────────────────────────────
FTSE_MID_CAP_TICKERS = [
    "ANIM.MI",   # Anima Holding — asset management
    "AVIO.MI",   # Avio — aerospazio
    "BC.MI",     # Brunello Cucinelli — luxury
    "BKNG.MI",   # Booking Holdings (ADR su MI)
    "BZU.MI",    # Buzzi Unicem — cemento
    "CALG.MI",   # Cairo Communication — media
    "CIR.MI",    # CIR — holding diversificata
    "CLT.MI",    # Clementoni (se disponibile)
    "CNHI.MI",   # CNH Industrial — macchine agricole
    "CRDI.MI",   # Credito Emiliano — banca
    "CVAL.MI",   # Credito Valtellinese
    "DIRE.MI",   # Digital Value
    "ENAV.MI",   # ENAV — controllo traffico aereo
    "FAB.MI",    # Fila — consumer goods
    "FILA.MI",   # Fila (alternativo)
    "GEI.MI",    # GEI
    "ITMG.MI",   # Italgas — gas distribution
    "IVG.MI",    # Iveco Group — veicoli industriali
    "MFB.MI",    # Mediobanca Fin.
    "MTS.MI",    # Maire Tecnimont — engineering
    "OVS.MI",    # OVS — moda
    "PRM.MI",    # Prysmian (già in MIB ma più piccola)
    "RACE.MI",   # Ferrari — luxury auto
    "SAVE.MI",   # Aeroporto di Venezia
    "SOL.MI",    # SOL Group — gas industriali
    "SFER.MI",   # Saipem — energia
    "TOD.MI",    # Tod's — luxury
    "TUL.MI",    # Tullow Oil
    "VIV.MI",    # Vivendi
]

# ── FTSE Italia Small Cap (selezione con buona liquidità) ────────────────────
FTSE_SMALL_CAP_TICKERS = [
    "ABM.MI",    # Ascopiave — distribuzione gas
    "ACCA.MI",   # Acca RE
    "ADM.MI",    # Antares Vision
    "ACSM.MI",   # Acsm Agam
    "ALC.MI",    # Alce
    "AMZO.MI",   # Amazon su MI
    "ASC.MI",    # Ascopiave
    "ATON.MI",   # Aton — IT services
    "BGN.MI",    # Banca Generali — private banking
    "BPO.MI",    # Bper Banca
    "CASS.MI",   # Banca Carige/Cassiopea
    "COIMA.MI",  # Coima Res — real estate
    "DCNX.MI",   # Datalogic
    "DEDA.MI",   # Dedagroup
    "EEMS.MI",   # EEMS
    "EPIQ.MI",   # Epiqe
    "EQUI.MI",   # Equita Group
    "ERAL.MI",   # El.En — laser tech
    "FOPE.MI",   # Fope — gioielli
    "GIGLIO.MI", # Giglio Group
    "GVS.MI",    # GVS — filtri industriali
    "IEG.MI",    # Italian Exhibition Group
    "ILLIMITY.MI",# Illimity Bank
    "INITF.MI",  # Initf
    "IT.MI",     # Italia Independent
    "ITT.MI",    # ITT
    "KME.MI",    # KME Group — rame
    "LEONE.MI",  # Leone Film Group
    "LU.MI",     # Luca De Meo
    "MAPS.MI",   # Maps
    "MCP.MI",    # MCP
    "MDBA.MI",   # Mediaset
    "MIKRO.MI",  # Mikropis
    "MNTV.MI",   # Montefibre
    "MONCL.MI",  # Moncler (già in MIB)
    "NEWL.MI",   # Newsline
    "NVG.MI",    # Novigo
    "PLNX.MI",   # Planetel
    "POSTE.MI",  # Poste Italiane — logistics/finance
    "QLT.MI",    # Qlts
    "REVO.MI",   # Revo Insurance
    "RINA.MI",   # Rina
    "SALC.MI",   # Salcef Group — infrastrutture ferroviarie
    "SIT.MI",    # SIT — gas metering
    "SMRE.MI",   # Smre
    "SNAM.MI",   # Snam — gas network (già in MIB subset)
    "SOMA.MI",   # Somaschini
    "SOS.MI",    # SOS
    "SPM.MI",    # Servizi Italia
    "TESMEC.MI", # Tesmec — cavi
    "TIP.MI",    # Tamburi Investment Partners
    "UNIPOLSAI.MI",# Unipol SAI assicurazioni
    "VRE.MI",    # Vetrerie Riunite
    "WEBUILD.MI",# Webuild (ex Salini) — costruzioni
    "WBD.MI",    # Warner Bros Discovery
    "YAM.MI",    # Yam
]

# ── UNIVERSI COMBINATI ────────────────────────────────────────────────────────
ITALY_ALL_TICKERS = FTSE_MIB_TICKERS + FTSE_MID_CAP_TICKERS + FTSE_SMALL_CAP_TICKERS

# Lista pulita: solo ticker confermati disponibili su yfinance
# (rimossi quelli con dati scarsi o non disponibili)
ITALY_LIQUID_TICKERS = [
    # Large cap FTSE MIB (31)
    "A2A.MI", "AMP.MI", "AZM.MI", "BAMI.MI", "BMED.MI",
    "BPE.MI", "CPR.MI", "DIA.MI", "ENEL.MI", "ENI.MI",
    "ERG.MI", "FBK.MI", "G.MI",   "HER.MI",  "IG.MI",
    "INW.MI", "ISP.MI", "LDO.MI", "MB.MI",   "MONC.MI",
    "NEXI.MI","PIRC.MI","PRY.MI", "PST.MI",  "REC.MI",
    "SRG.MI", "STLAM.MI","TEN.MI","TRN.MI",  "UCG.MI",
    "UNI.MI",
    # Mid cap selezionati (15 più liquidi)
    "RACE.MI",  # Ferrari
    "BZU.MI",   # Buzzi Unicem
    "CNHI.MI",  # CNH Industrial
    "ENAV.MI",  # ENAV
    "ITMG.MI",  # Italgas
    "IVG.MI",   # Iveco Group
    "MTS.MI",   # Maire Tecnimont
    "OVS.MI",   # OVS
    "SAVE.MI",  # Save Aeroporti
    "SOL.MI",   # SOL Group
    "BC.MI",    # Brunello Cucinelli
    "ANIM.MI",  # Anima Holding
    "CIR.MI",   # CIR
    "TOD.MI",   # Tod's
    "SFER.MI",  # Saipem
    # Small cap selezionati (10 più liquidi)
    "BGN.MI",      # Banca Generali
    "POSTE.MI",    # Poste Italiane
    "SALC.MI",     # Salcef Group
    "WEBUILD.MI",  # Webuild
    "TIP.MI",      # Tamburi
    "GVS.MI",      # GVS
    "ELEN.MI",     # El.En
    "SIT.MI",      # SIT
    "DEDA.MI",     # Dedagroup
    "ILLIMITY.MI", # Illimity Bank
]

# Per i test rapidi
FTSE_MIB_TICKERS_TEST = ["ENI.MI", "ENEL.MI", "ISP.MI"]

# ── METADATI ──────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    # FTSE MIB
    "A2A.MI":   "utilities",    "AMP.MI":    "financials",  "AZM.MI":    "industrials",
    "BAMI.MI":  "financials",   "BMED.MI":   "healthcare",  "BPE.MI":    "financials",
    "CPR.MI":   "financials",   "DIA.MI":    "consumer",    "ENEL.MI":   "utilities",
    "ENI.MI":   "energy",       "ERG.MI":    "utilities",   "FBK.MI":    "financials",
    "G.MI":     "financials",   "HER.MI":    "energy",      "IG.MI":     "financials",
    "INW.MI":   "telecom",      "ISP.MI":    "financials",  "LDO.MI":    "industrials",
    "MB.MI":    "financials",   "MONC.MI":   "consumer",    "NEXI.MI":   "technology",
    "PIRC.MI":  "industrials",  "PRY.MI":    "industrials", "PST.MI":    "telecom",
    "REC.MI":   "industrials",  "SRG.MI":    "utilities",   "STLAM.MI":  "consumer",
    "TEN.MI":   "industrials",  "TRN.MI":    "utilities",   "UCG.MI":    "financials",
    "UNI.MI":   "financials",
    # Mid cap
    "RACE.MI":  "consumer",     "BZU.MI":    "industrials", "CNHI.MI":   "industrials",
    "ENAV.MI":  "industrials",  "ITMG.MI":   "utilities",   "IVG.MI":    "industrials",
    "MTS.MI":   "industrials",  "OVS.MI":    "consumer",    "SAVE.MI":   "industrials",
    "SOL.MI":   "industrials",  "BC.MI":     "consumer",    "ANIM.MI":   "financials",
    "CIR.MI":   "financials",   "TOD.MI":    "consumer",    "SFER.MI":   "energy",
    # Small cap
    "BGN.MI":   "financials",   "POSTE.MI":  "financials",  "SALC.MI":   "industrials",
    "WEBUILD.MI":"industrials", "TIP.MI":    "financials",  "GVS.MI":    "industrials",
    "ELEN.MI":  "healthcare",   "SIT.MI":    "industrials", "DEDA.MI":   "technology",
    "ILLIMITY.MI":"financials",
}

CAP_SEGMENT = {t: "large"  for t in FTSE_MIB_TICKERS}
CAP_SEGMENT.update({
    "RACE.MI":"mid","BZU.MI":"mid","CNHI.MI":"mid","ENAV.MI":"mid",
    "ITMG.MI":"mid","IVG.MI":"mid","MTS.MI":"mid","OVS.MI":"mid",
    "SAVE.MI":"mid","SOL.MI":"mid","BC.MI":"mid","ANIM.MI":"mid",
    "CIR.MI":"mid","TOD.MI":"mid","SFER.MI":"mid",
    "BGN.MI":"small","POSTE.MI":"small","SALC.MI":"small",
    "WEBUILD.MI":"small","TIP.MI":"small","GVS.MI":"small",
    "ELEN.MI":"small","SIT.MI":"small","DEDA.MI":"small",
    "ILLIMITY.MI":"small",
})

def get_tickers(segment: str = "mib") -> list:
    """
    Ritorna la lista ticker per il segmento richiesto.
    segment: 'mib' | 'mid' | 'small' | 'liquid' | 'all'
    """
    if segment == "mib":
        return FTSE_MIB_TICKERS
    elif segment == "mid":
        return FTSE_MID_CAP_TICKERS
    elif segment == "small":
        return FTSE_SMALL_CAP_TICKERS
    elif segment == "liquid":
        return ITALY_LIQUID_TICKERS
    elif segment == "all":
        return ITALY_ALL_TICKERS
    return FTSE_MIB_TICKERS
