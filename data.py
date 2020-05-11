#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 09:25:51 2019

@author: daniele
"""
#%%
def getInitTeams(teams):
    
    initTeams = ''
    
    for team in teams:
        
        initTeams += team.upper() + ' = \'' + team + '\'' + '\n'
        
    return initTeams

def getUpperTeams(teams):
    
    res_teams = ''
    
    for team in teams:
        res_teams += team.upper() + ', '
        
    return res_teams
#%%
SERIE_A = 'Serie_A'
SERIE_B = 'Serie_B'
JUPILIER = 'Jupilier'
LIGUE_1 = 'Ligue_1'
LIGUE_2 = 'Ligue_2'
LIGA = 'Liga'
LIGA_2 = 'Liga_2'
BUNDESLIGA = 'Bundesliga'
BUNDESLIGA_2 = 'Bundesliga_2'
PREMIER = 'Premier'
PREMIER_2 = 'Premier_2'
EREDIVISIE = 'Eredivisie'

LEAGUE_NAMES = [SERIE_A, JUPILIER, LIGUE_1, LIGUE_2, LIGA, LIGA_2, BUNDESLIGA, BUNDESLIGA_2, PREMIER, PREMIER_2, EREDIVISIE]

#%%
#serie_A = ['Data/Italy/Serie_A/I1_14-15.csv',
#           'Data/Italy/Serie_A/I1_15-16.csv',
#           'Data/Italy/Serie_A/I1_16-17.csv',
#           'Data/Italy/Serie_A/I1_17-18.csv',
#           'Data/Italy/Serie_A/I1_18-19.csv']             
#serieA_19 = ['Data/Italy/Serie_A/I1_18-19.csv']
#serieA_18 = 'Data/Italy/Serie_A/I1_17-18.csv'
#serieA_17 = 'Data/Italy/Serie_A/I1_16-17.csv'
#serieA_16 = 'Data/Italy/Serie_A/I1_15-16.csv'
#serieA_15 = 'Data/Italy/Serie_A/I1_14-15.csv'
serie_A = ['https://www.football-data.co.uk/mmz4281/1920/I1.csv',
             'https://www.football-data.co.uk/mmz4281/1819/I1.csv',
             'https://www.football-data.co.uk/mmz4281/1718/I1.csv',
             'https://www.football-data.co.uk/mmz4281/1617/I1.csv',
             'https://www.football-data.co.uk/mmz4281/1516/I1.csv',
             'https://www.football-data.co.uk/mmz4281/1415/I1.csv']

serie_B = ['https://www.football-data.co.uk/mmz4281/1920/I2.csv',
           'https://www.football-data.co.uk/mmz4281/1819/I2.csv',
           'https://www.football-data.co.uk/mmz4281/1718/I2.csv',
           'https://www.football-data.co.uk/mmz4281/1617/I2.csv',
           'https://www.football-data.co.uk/mmz4281/1516/I2.csv',
           'https://www.football-data.co.uk/mmz4281/1415/I2.csv']

jupilier_belgio = ['https://www.football-data.co.uk/mmz4281/1920/B1.csv',
                   'https://www.football-data.co.uk/mmz4281/1819/B1.csv',
                   'https://www.football-data.co.uk/mmz4281/1718/B1.csv',
                   'https://www.football-data.co.uk/mmz4281/1617/B1.csv',
                   'https://www.football-data.co.uk/mmz4281/1516/B1.csv',
                   'https://www.football-data.co.uk/mmz4281/1415/B1.csv']

ligue_1 = ['https://www.football-data.co.uk/mmz4281/1920/F1.csv',
           'https://www.football-data.co.uk/mmz4281/1819/F1.csv',
           'https://www.football-data.co.uk/mmz4281/1718/F1.csv',
           'https://www.football-data.co.uk/mmz4281/1617/F1.csv',
           'https://www.football-data.co.uk/mmz4281/1516/F1.csv',
           'https://www.football-data.co.uk/mmz4281/1415/F1.csv']

ligue_2 = ['https://www.football-data.co.uk/mmz4281/1920/F2.csv',
         'https://www.football-data.co.uk/mmz4281/1819/F2.csv',
         'https://www.football-data.co.uk/mmz4281/1718/F2.csv',
         'https://www.football-data.co.uk/mmz4281/1617/F2.csv',
         'https://www.football-data.co.uk/mmz4281/1516/F2.csv',
         'https://www.football-data.co.uk/mmz4281/1415/F2.csv']

liga = ['https://www.football-data.co.uk/mmz4281/1920/SP1.csv',
        'https://www.football-data.co.uk/mmz4281/1819/SP1.csv',
        'https://www.football-data.co.uk/mmz4281/1718/SP1.csv',
        'https://www.football-data.co.uk/mmz4281/1617/SP1.csv',
        'https://www.football-data.co.uk/mmz4281/1516/SP1.csv',
        'https://www.football-data.co.uk/mmz4281/1415/SP1.csv']

liga_2 = ['https://www.football-data.co.uk/mmz4281/1920/SP2.csv',
          'https://www.football-data.co.uk/mmz4281/1819/SP2.csv',
          'https://www.football-data.co.uk/mmz4281/1718/SP2.csv',
          'https://www.football-data.co.uk/mmz4281/1617/SP2.csv',
          'https://www.football-data.co.uk/mmz4281/1516/SP2.csv',
          'https://www.football-data.co.uk/mmz4281/1415/SP2.csv']

bundesliga = ['https://www.football-data.co.uk/mmz4281/1920/D1.csv',
              'https://www.football-data.co.uk/mmz4281/1819/D1.csv',
              'https://www.football-data.co.uk/mmz4281/1718/D1.csv',
              'https://www.football-data.co.uk/mmz4281/1617/D1.csv',
              'https://www.football-data.co.uk/mmz4281/1516/D1.csv',
              'https://www.football-data.co.uk/mmz4281/1415/D1.csv']

bundesliga_2 = ['https://www.football-data.co.uk/mmz4281/1920/D2.csv',
                'https://www.football-data.co.uk/mmz4281/1819/D2.csv',
                'https://www.football-data.co.uk/mmz4281/1718/D2.csv',
                'https://www.football-data.co.uk/mmz4281/1617/D2.csv',
                'https://www.football-data.co.uk/mmz4281/1516/D2.csv',
                'https://www.football-data.co.uk/mmz4281/1415/D2.csv']

premier = ['https://www.football-data.co.uk/mmz4281/1920/E0.csv',
           'https://www.football-data.co.uk/mmz4281/1819/E0.csv',
           'https://www.football-data.co.uk/mmz4281/1718/E0.csv',
           'https://www.football-data.co.uk/mmz4281/1617/E0.csv',
           'https://www.football-data.co.uk/mmz4281/1516/E0.csv',
           'https://www.football-data.co.uk/mmz4281/1415/E0.csv']

premier_2 = ['https://www.football-data.co.uk/mmz4281/1920/E1.csv',
             'https://www.football-data.co.uk/mmz4281/1819/E1.csv',
             'https://www.football-data.co.uk/mmz4281/1718/E1.csv',
             'https://www.football-data.co.uk/mmz4281/1617/E1.csv',
             'https://www.football-data.co.uk/mmz4281/1516/E1.csv',
             'https://www.football-data.co.uk/mmz4281/1415/E1.csv']

eredivisie = ['https://www.football-data.co.uk/mmz4281/1920/N1.csv',
          'https://www.football-data.co.uk/mmz4281/1819/N1.csv',
          'https://www.football-data.co.uk/mmz4281/1718/N1.csv',
          'https://www.football-data.co.uk/mmz4281/1617/N1.csv',
          'https://www.football-data.co.uk/mmz4281/1516/N1.csv',
          'https://www.football-data.co.uk/mmz4281/1415/N1.csv']

LEAGUE_CSV = {SERIE_A : serie_A,
                JUPILIER: jupilier_belgio,
                LIGUE_1: ligue_1,
                LIGUE_2:ligue_2,
                LIGA:liga,
                LIGA_2:liga_2,
                BUNDESLIGA:bundesliga,
                BUNDESLIGA_2:bundesliga_2,
                PREMIER:premier,
                PREMIER_2:premier_2,
                EREDIVISIE:eredivisie}
#%% SERIE A
PARMA = 'Parma'
FIORENTINA = 'Fiorentina'
UDINESE = 'Udinese'
CAGLIARI = 'Cagliari'
ROMA = 'Roma'
SAMPDORIA = 'Sampdoria'
SPAL = 'Spal'
TORINO = 'Torino'
VERONA = 'Verona'
INTER = 'Inter'
BOLOGNA = 'Bologna'
MILAN = 'Milan'
JUVENTUS = 'Juventus'
LAZIO = 'Lazio'
ATALANTA = 'Atalanta'
GENOA = 'Genoa'
LECCE = 'Lecce'
SASSUOLO = 'Sassuolo'
NAPOLI = 'Napoli'
BRESCIA = 'Brescia'

TEAMS_SERIE_A = [PARMA, FIORENTINA, UDINESE, CAGLIARI, ROMA, SAMPDORIA, SPAL, TORINO, VERONA, INTER, BOLOGNA, MILAN, JUVENTUS,
                 LAZIO, ATALANTA, GENOA, LECCE, SASSUOLO, NAPOLI, BRESCIA]
#%% SERIE B

#%% LIGA
ATH_BILBAO = 'Ath Bilbao'
CELTA = 'Celta'
VALENCIA = 'Valencia'
MALLORCA = 'Mallorca'
LEGANES = 'Leganes'
VILLARREAL = 'Villarreal'
ALAVES = 'Alaves'
ESPANOL = 'Espanol'
BETIS = 'Betis'
ATH_MADRID = 'Ath Madrid'
GRANADA = 'Granada'
LEVANTE = 'Levante'
OSASUNA = 'Osasuna'
REAL_MADRID = 'Real Madrid'
GETAFE = 'Getafe'
BARCELLONA = 'Barcelona'
SEVILLA = 'Sevilla'
REAL_SOCIEDAD = 'Sociedad'
EIBAR = 'Eibar'
VALLADOLID = 'Valladolid'

TEAMS_LIGA = [ATH_BILBAO, ATH_MADRID, VALENCIA, MALLORCA, CELTA, LEGANES, VILLARREAL, ALAVES, ESPANOL, BETIS, GRANADA, LEVANTE, OSASUNA,
              REAL_MADRID, REAL_SOCIEDAD, GETAFE, BARCELLONA, SEVILLA, EIBAR, VALLADOLID]

#%% LIGA_2
LUGO = 'Lugo'
SANTANDER = 'Santander'
ALMERIA = 'Almeria'
ELCHE = 'Elche'
VALLECANO = 'Vallecano'
SARAGOZZA = 'Zaragoza'
DEP_LA_CORUNA = 'La Coruna'
NUMANCIA = 'Numancia'
GIRONA = 'Girona' 
CADIZ = 'Cadiz'
LAS_PALMAS = 'Las Palmas'
ALBACETE = 'Albacete'
OVIEDO = 'Oviedo'
MIRANDES = 'Mirandes'
ALCORON = 'Alcorcon'
MALAGA = 'Malaga'
PONFERRADINA = 'Ponferradina'
GIJON = 'Sp Gijon'
HUESCA = 'Huesca'
TENERIFE = 'Tenerife'
EXTREMADURA = 'Extremadura UD'
FUENLABRADA = 'Fuenlabrada'
RACING_SANTANDER = 'Racing Santander'

TEAMS_LIGA_2 = [LUGO, SANTANDER, ALMERIA, ELCHE, VALLECANO, SARAGOZZA, DEP_LA_CORUNA, NUMANCIA, GIRONA, CADIZ, LAS_PALMAS, ALBACETE,
                OVIEDO, MIRANDES, ALCORON, MALAGA, PONFERRADINA, GIJON, HUESCA, TENERIFE, EXTREMADURA, FUENLABRADA, RACING_SANTANDER]
#%% JUPILIER
GENK = 'Genk'
CERCLE_BRUGGE = 'Cercle Brugge'
ST_TRUIDEN = 'St Truiden'
WAREGEM = 'Waregem'
WAASLAND_BEVEREN = 'Waasland-Beveren'
ANDERLECHT = 'Anderlecht'
CHARLEROI = 'Charleroi'
EUPEN = 'Eupen'
CLUB_BRUGGE = 'Club Brugge'
ST_LIEGI = 'Standard'
KORTRIJK = 'Kortrijk'
OOSTENDE = 'Oostende'
MECHELEN = 'Mechelen'
GENT = 'Gent'
MOUSCRON = 'Mouscron'
ANTWERP = 'Antwerp'

TEAMS_JUPILIER = [GENK, CERCLE_BRUGGE, ST_TRUIDEN, WAREGEM, WAASLAND_BEVEREN, ANDERLECHT, CHARLEROI, EUPEN, CLUB_BRUGGE, ST_LIEGI, 
                  KORTRIJK, OOSTENDE, MECHELEN, GENT, MOUSCRON, ANTWERP]
#%% LIGUE_1
MONACO = 'Monaco'
MARSEILLE = 'Marseille'
ANGERS = 'Angers'
BREST = 'Brest'
DIJON = 'Dijon'
MONTPELLIER = 'Montpellier'
NICE = 'Nice'
LILLE = 'Lille'
STRASBOURG = 'Strasbourg'
PARIS_SG = 'Paris SG'
LYON = 'Lyon'
NANTES = 'Nantes'
AMIENS = 'Amiens'
BORDEAUX = 'Bordeaux'
METZ = 'Metz'
NIMES = 'Nimes'
TOULOUSE = 'Toulouse'
ST_ETIENNE = 'St Etienne'
REIMS = 'Reims'
RENNES = 'Rennes'

TEAMS_LIGUE_1 = [MONACO, MARSEILLE, ANGERS, BREST, DIJON, MONTPELLIER, NICE, LILLE, STRASBOURG, PARIS_SG, LYON, NANTES, AMIENS, BORDEAUX,
                 METZ, NIMES, TOULOUSE, ST_ETIENNE, REIMS, RENNES]
#%% LIGUE 2
AJACCIO = 'Ajaccio'
CHAMBLY = 'Chambly'
CLERMONT = 'Clermont'
GUINGAMP = 'Guingamp'
NANCY = 'Nancy'
NIORT = 'Niort'
RODEZ = 'Rodez'
SOCHAUX = 'Sochaux'
LE_MANS = 'Le Mans'
LORIENT = 'Lorient'
AUXERRE = 'Auxerre'
CHATEAUROUX = 'Chateauroux'
GRENOBLE = 'Grenoble'
LE_HAVRE = 'Le Havre'
ORLEANS = 'Orleans'
PARIS_FC = 'Paris FC'
TROYES = 'Troyes'
VALENCIENNES = 'Valenciennes'
LENS = 'Lens'
CAEN = 'Caen'

TEAMS_LIGUE_2 = [AJACCIO, CHAMBLY, CLERMONT, GUINGAMP, NANCY, NIORT, RODEZ, SOCHAUX, LE_MANS, LORIENT, AUXERRE, CHATEAUROUX, GRENOBLE,
                  LE_HAVRE, ORLEANS, PARIS_FC, TROYES, VALENCIENNES, LENS, CAEN]
#%% BUNDESLIGA 
BAYERN_MUNICH = 'Bayern Munich'
DORTMUND = 'Dortmund'
FREIBURG = 'Freiburg'
LEVERKUSEN = 'Leverkusen'
WERDER_BREMEN = 'Werder Bremen'
WOLFSBURG = 'Wolfsburg'
M_GLADBACH = 'M\'gladbach'
EIN_FRANKFURT = 'Ein Frankfurt'
UNION_BERLIN = 'Union Berlin'
FC_KOLN = 'FC Koln'
AUGSBURG = 'Augsburg'
FORTUNA_DUSSELDORF = 'Fortuna Dusseldorf'
HOFFENHEIM = 'Hoffenheim'
MAINZ = 'Mainz'
PADERBORN = 'Paderborn'
SCHALKE_04 = 'Schalke 04'
RB_LEIPZIG = 'RB Leipzig'
HERTHA = 'Hertha'

TEAMS_BUNDESLIGA = [BAYERN_MUNICH, DORTMUND, FREIBURG, LEVERKUSEN, WERDER_BREMEN, WOLFSBURG, M_GLADBACH, EIN_FRANKFURT,
                    UNION_BERLIN, FC_KOLN, AUGSBURG, FORTUNA_DUSSELDORF, HOFFENHEIM, MAINZ, PADERBORN, SCHALKE_04, RB_LEIPZIG, HERTHA]
#%% BUNDESLIGA 2
STUTTGART = 'Stuttgart'
DRESDEN = 'Dresden'
HOLSTEIN_KIEL = 'Holstein Kiel'
OSNABRUCK = 'Osnabruck'
HAMBURG = 'Hamburg'
GREUTHER_FURTH = 'Greuther Furth'
REGENSBURG = 'Regensburg'
WEHEN = 'Wehen'
BIELEFELD = 'Bielefeld'
BOCHUM = 'Bochum'
SANDHAUSEN = 'Sandhausen'
ST_PAULI = 'St Pauli'
KARLSRUHE = 'Karlsruhe'
HANNOVER = 'Hannover'
HEIDENHEIM = 'Heidenheim'
DARMSTADT = 'Darmstadt'
ERZGEBIRGE_AUE = 'Erzgebirge Aue'
NORIMBERGA = 'Nurnberg'

TEAMS_BUNDESLIGA_2 = [STUTTGART, DRESDEN, HOLSTEIN_KIEL, OSNABRUCK, HAMBURG, GREUTHER_FURTH, REGENSBURG,
                      WEHEN, BIELEFELD, BOCHUM, SANDHAUSEN, ST_PAULI, KARLSRUHE, HANNOVER, HEIDENHEIM, DARMSTADT, ERZGEBIRGE_AUE, NORIMBERGA]
#%% PREMIER
LIVERPOOL = 'Liverpool'
WEST_HAM = 'West Ham'
BOURNEMOUTH = 'Bournemouth'
BURNLEY = 'Burnley'
CRYSTAL_PALACE = 'Crystal Palace'
WATFORD = 'Watford'
TOTTENHAM = 'Tottenham'
LEICESTER = 'Leicester'
NEWCASTLE = 'Newcastle'
MAN_UNITED = 'Man United'
ARSENAL = 'Arsenal'
ASTON_VILLA = 'Aston Villa'
BRIGHTON = 'Brighton'
EVERTON = 'Everton'
NORWICH = 'Norwich'
SOUTHAMPTON = 'Southampton'
MAN_CITY = 'Man City'
SHEFFIELD_UNITED = 'Sheffield United'
CHELSEA = 'Chelsea'
WOLVES = 'Wolves'

TEAMS_PREMIER = [LIVERPOOL, WEST_HAM, BOURNEMOUTH, BURNLEY, CRYSTAL_PALACE, WATFORD, TOTTENHAM, LEICESTER, NEWCASTLE, MAN_UNITED,
                 ARSENAL, ASTON_VILLA, BRIGHTON, EVERTON, NORWICH, SOUTHAMPTON, MAN_CITY, SHEFFIELD_UNITED, CHELSEA, WOLVES]
#%% PREMIER 2

LUTON = 'Luton'
BARNSLEY = 'Barnsley'
BLACKBURN = 'Blackburn'
BRENTFORD = 'Brentford'
MILLWALL = 'Millwall'
READING = 'Reading'
STOKE = 'Stoke'
SWANSEA = 'Swansea'
WIGAN = 'Wigan'
NOTTINGAM_FOREST = 'Nott\'m Forest'
BRISTOL_CITY = 'Bristol City'
HUDDERSFIELD = 'Huddersfield'
LEEDS = 'Leeds'
BIRMINGHAM = 'Birmingham'
CARDIFF = 'Cardiff'
CHARLTON = 'Charlton'
DERBY = 'Derby'
FULHAM = 'Fulham'
HULL = 'Hull'
MIDDLESBROUGH = 'Middlesbrough'
PRESTON = 'Preston'
QPR = 'QPR'
SHEFFIELD_WEDS = 'Sheffield Weds'
WEST_BROM = 'West Brom'

TEAMS_PREMIER_2 = [LUTON, BARNSLEY, BLACKBURN, BRENTFORD, MILLWALL, READING, STOKE, SWANSEA, WIGAN, NOTTINGAM_FOREST, BRISTOL_CITY,
                   HUDDERSFIELD, LEEDS, BIRMINGHAM, CARDIFF, CHARLTON, DERBY, FULHAM, HULL, MIDDLESBROUGH, PRESTON, QPR, SHEFFIELD_WEDS,
                   WEST_BROM]

#%% EREDIVISIE

ZWOLLE = 'Zwolle'
FC_EMMEN = 'FC Emmen'
VITESSE = 'Vitesse'
TWENTE = 'Twente'
VVV_VENLO = 'VVV Venlo'
HERACLES = 'Heracles'
FEYENOORD = 'Feyenoord'
DEN_HAAG = 'Den Haag'
AZ_ALKMAAR = 'AZ Alkmaar'
SPARTA_ROTTERDAM = 'Sparta Rotterdam'
GRONINGEN = 'Groningen'
AJAX = 'Ajax'
WILLEM_II = 'Willem II'
FOR_SITTARD = 'For Sittard'
HEERENVEEN = 'Heerenveen'
WAALWIJK = 'Waalwijk'
UTRECHT = 'Utrecht'
PSV_EINDHOVEN = 'PSV Eindhoven'

TEAMS_EREDIVISIE = [ZWOLLE, FC_EMMEN, VITESSE, TWENTE, VVV_VENLO, HERACLES, FEYENOORD, DEN_HAAG, AZ_ALKMAAR,
                    SPARTA_ROTTERDAM, GRONINGEN, AJAX, WILLEM_II, FOR_SITTARD, HEERENVEEN, WAALWIJK, UTRECHT, PSV_EINDHOVEN]
#%%

TEAMS_LEAGUE = {SERIE_A : TEAMS_SERIE_A,
                JUPILIER: TEAMS_JUPILIER,
                LIGUE_1: TEAMS_LIGUE_1,
                LIGUE_2:TEAMS_LIGUE_2,
                LIGA:TEAMS_LIGA,
                LIGA_2:TEAMS_LIGA_2,
                BUNDESLIGA:TEAMS_BUNDESLIGA,
                BUNDESLIGA_2:TEAMS_BUNDESLIGA_2,
                PREMIER:TEAMS_PREMIER,
                PREMIER_2:TEAMS_PREMIER_2,
                EREDIVISIE:TEAMS_EREDIVISIE}