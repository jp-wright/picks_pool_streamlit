""" 
refs
"""
from utils.constants import CURR_SEASON
POOL_YEARS = list(range(2017, CURR_SEASON + 1))
POOL_YEARS.remove(2022)  ## no pool in 2022

conf_dct = {
    'Chiefs': 'AFC',
    'Bills': 'AFC',
    'Patriots': 'AFC',
    'Browns': 'AFC',
    'Ravens': 'AFC',
    'Titans': 'AFC',
    'Chargers': 'AFC',
    'Colts': 'AFC',
    'Dolphins': 'AFC',
    'Broncos': 'AFC',
    'Steelers': 'AFC',
    'Jets': 'AFC',
    'Raiders': 'AFC',
    'Jaguars': 'AFC',
    'Bengals': 'AFC',
    'Texans': 'AFC',
    'Buccaneers': 'NFC',
    'Rams': 'NFC',
    'Packers': 'NFC',
    '49ers': 'NFC',
    'Seahawks': 'NFC',
    'Cowboys': 'NFC',
    'Saints': 'NFC',
    'Falcons': 'NFC',
    'Vikings': 'NFC',
    'Redskins': 'NFC',
    'Commanders': 'NFC',
    'Football Team': 'NFC',
    'Cardinals': 'NFC',
    'Bears': 'NFC',
    'Giants': 'NFC',
    'Panthers': 'NFC',
    'Eagles': 'NFC',
    'Lions': 'NFC',
    }

champ_hist = {
    2017: 'Jackson',
    2018: 'Brandon',
    2019: 'Jordan',
    2020: 'Alex',
    2021: 'Dan',
    2022: 'NoPool',
    2023: 'Victoria',
    }


GAMES_PER_SEASON = {
    2017: 16,
    2018: 16,
    2019: 16,
    2020: 16,
    2021: 17,
    2022: 17,
    2023: 17,
    2024: 17,
    2025: 17,
    2026: 17,
}