import azure.functions as func
import datetime
import json
import logging
import requests
import pandas as pd
import numpy as np
import lxml.html as lh
import re

app = func.FunctionApp()

@app.function_name(name="data_acquisition")
@app.route(route="myroute", auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    
    def get_data(url, x_path):
        # Send a GET request to the webpage
        response = requests.get(url)

        # Parse the webpage content
        tree = lh.fromstring(response.content)

        # Extract the table using XPath
        table = tree.xpath(f'{x_path}')[0]

        # Convert the table to a pandas DataFrame
        df = pd.read_html(lh.tostring(table, encoding='unicode'))[0]
        return df

    #Acquire point totals by team and position

    pts_by_team_and_pos = get_data("https://www.fleaflicker.com/mlb/leagues/25375/leaders", '//*[@id="body-center-main"]/table')
    pts_by_team_and_pos = pts_by_team_and_pos.drop(columns=['#'])
    pts_by_team_and_pos[['Optimum PF', 'Efficiency']] = pts_by_team_and_pos["Optimum PF"].str.split('(', n = 1, expand = True)
    pts_by_team_and_pos['Efficiency'] = pts_by_team_and_pos['Efficiency'].str.replace(')', '')
    pts_by_team_and_pos['Efficiency'] = pts_by_team_and_pos['Efficiency'].str.replace('%', '')
    pts_by_team_and_pos['Efficiency'] = round(pts_by_team_and_pos['Efficiency'].astype(float) / 100, 2)


    def extract_characters(text):
        """
        Extracts the character immediately to the left of the period and everything to its right
        using regular expressions. If no such pattern is found, it returns the original text.

        Args:
            text: The input string.

        Returns:
            The modified string or the original string if no period with a preceding character is found.
        """
        match = re.search(r"(.{1})\.", text)
        if match:
            return match.group(1) + text[match.end(0) - 1:]
        else:
            return text

    # Make the column headers the first row and move everything else down
    def process_rosters(roster_df):  
        roster_df.index = roster_df.index + 1  # Shift the index
        roster_df.loc[0] = roster_df.columns[0]  # Add the column headers as a new row
        roster_df = roster_df.sort_index()    # Sort the index to reorder

        # Rename the columns
        roster_df.columns = ["Team", "Player", "NA"]

        roster_df.iloc[0, 1] = None
        roster_df = roster_df.dropna(how='all')  # Drop rows where all elements are NaN
        del roster_df['NA']
        roster_df['Team'] = roster_df['Team'].apply(lambda x: extract_characters(x))
        roster_df = roster_df[~roster_df['Team'].str.contains("Total", na=False)]
        roster_df.loc[roster_df['Team'].str.contains(r'\.', na=False), ['Player', 'Team']] = roster_df.loc[roster_df['Team'].str.contains(r'\.', na=False), ['Team', 'Player']].values
        roster_df['Team'] = roster_df['Team'].fillna(method='ffill')
        roster_df = roster_df[roster_df['Team'] != roster_df['Player']]
        roster_df = roster_df.dropna()
        roster_df['Player'] = roster_df['Player'].str.replace(r'(?<=\b\w\.\s)([A-Z][a-z]+)\s([A-Z][a-z]+)', r'\1\2', regex=True)
        roster_df[['Player', 'Position', 'MLB Team']] = roster_df['Player'].str.extract(r'([^ ]+ [^ ]+) ([^ ]+) (.+)')
        return roster_df

    #Acquire rosters

    rosters1 = pd.DataFrame(get_data("https://www.fleaflicker.com/mlb/leagues/25375/teams", '//*[@id="body-center-main"]/div/div[1]/table'))
    rosters2 = pd.DataFrame(get_data("https://www.fleaflicker.com/mlb/leagues/25375/teams", '//*[@id="body-center-main"]/div/div[2]/table'))
    rosters1 = process_rosters(rosters1)
    rosters2 = process_rosters(rosters2)
    rosters = pd.concat([rosters1, rosters2], ignore_index=True)

    standings = get_data('https://www.fleaflicker.com/mlb/leagues/25375','//*[@id="body-center-main"]/table')
    standings.columns = ['NA1',
                        'NA2',
                        'Team',
                        'Owner',
                        'NA3',
                        'Wins',
                        'Losses',
                        'Win Percentage',
                        'GB_Div',
                        'Div Record',
                        'GB_Playoffs',
                        'Strk',
                        'NA4',
                        'PF',
                        'PF Avg',
                        'PA',
                        'PA Avg',
                        'NA5',
                        'Waiver Priority',
                        'Rank',
                        'NA6',
                        'NA7',
                        'NA8']
    standings = standings.loc[:, ~standings.columns.str.contains('^NA')]
    standings = standings.dropna(how='all')
    standings = standings.loc[~standings.eq(standings.iloc[:, 0], axis=0).all(axis=1)]
    standings['Division'] = None
    standings = standings.reset_index(drop=True)
    standings['Division'].iloc[0:5] = 'Division 1' 
    standings['Division'].iloc[5:10] = 'Division 2'

    roster_urls = {'crippled_gang' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142196?statType=1',
                'gary_and_the_whales' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142102?statType=1',
                'skenes_world' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142197?statType=1',
                'redshift' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142199?statType=1',
                'roki_dokey_artichokey' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142207?statType=1',
                'rookers_and_blow' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142213?statType=1',
                'sisters_of_the_poor' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142214?statType=1',
                'latina_turner' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142296?statType=1',
                'du_yarvish' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142307?statType=1',
                'sinister_slingers' : 'https://www.fleaflicker.com/mlb/leagues/25375/teams/142327?statType=1'}
    def get_team_roster(url, xpath):
        roster = get_data(url, xpath)
        return roster
    
    team_rosters = {}
    for team, url in roster_urls.items():
        team_rosters[team] = get_team_roster(url, '//*[@id="body-center-main"]/table')
        team_rosters[team].name = team
    

    hitter_cols = ['Player',
                    'NA1',
                    'NA2',
                    'NA3',
                    'NA4',
                    'Avg',
                    'HR',
                    'R',
                    'RBI',
                    'SB',
                    'BB',
                    'SO',
                    'FPts',
                    'Avg FPts',
                    'NA5',
                    'NA6',
                    'Pos',
                    'Team'
                    ]

    pitcher_cols = ['Player',
                    'NA1',
                    'NA2',
                    'NA3',
                    'NA4',
                    'IP',
                    'BB',
                    'SO',
                    'W',
                    'SV',
                    'ERA',
                    'WHIP',
                    'FPts',
                    'Avg FPts',
                    'NA5',
                    'NA6',
                    'Pos',
                    'Team'
                    ]
    pitcher_pos = ['SP','RP','P']
    hitter_pos = ['C','1B','2B','3B','SS','OF','DH','LF','CF','RF']

    def adjust_pitcher_data(df):
        """
        Adjusts the data for rows where the Role is 'Pitcher' by moving hitter stats to pitcher stats columns.

        Args:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The adjusted DataFrame.
        """
        pitcher_mask = df['Role'] == 'Pitcher'
        columns_to_move = ['Avg', 'HR', 'R', 'RBI', 'SB', 'BB_Batting', 'SO_Batting']
        target_columns = ['IP', 'BB_Pitching', 'SO_Pitching', 'W', 'SV', 'ERA', 'WHIP']
        df.loc[pitcher_mask, target_columns] = df.loc[pitcher_mask, columns_to_move].values
        df.loc[pitcher_mask, columns_to_move] = None
        return df

    def adjust_hitter_data(df):
        """
        Adjusts the data for rows where the Role is 'Hitter' by moving pitcher stats to hitter stats columns.

        Args:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The adjusted DataFrame.
        """
        hitter_mask = df['Role'] == 'Hitter'
        target_columns = ['Avg', 'HR', 'R', 'RBI', 'SB', 'BB_Batting', 'SO_Batting']
        columns_to_move = ['IP', 'BB_Pitching', 'SO_Pitching', 'W', 'SV', 'ERA', 'WHIP']
        df.loc[hitter_mask, target_columns] = df.loc[hitter_mask, columns_to_move].values
        df.loc[hitter_mask, columns_to_move] = None
        return df

    def process_roster(df):    
        df.columns = [f"A{i+1}" for i in range(len(df.columns))]
        df = df.dropna(subset = ['A1'])
        df = df[df['A2'].isna()]
        df['A1'] = df['A1'].apply(lambda x: re.sub(r'\b(?:7D|10D|15D|30D|60D|OUT|DTD)\b', '', x).strip() if isinstance(x, str) else x)
        df['A1'] = df['A1'].str.lstrip('0123456789D').str.strip() if df['A1'].dtype == 'object' else df['A1']
        df['A1'] = df['A1'].str.replace('OUT', '', regex=False).str.strip() if df['A1'].dtype == 'object' else df['A1']
        df[['Player', 'Position', 'MLB Team']] = df['A1'].str.extract(r'([^ ]+ [^ ]+) ([^ ]+) (.+)')
        del df['A1']
        df['Role'] = df['Position'].apply(
            lambda pos: "Pitcher" if any(pitcher in pos for pitcher in pitcher_pos) else "Hitter"
        )
        df = pd.concat([df, df.loc[:, 'A6':'A12'].add_suffix('_extra')], axis=1)
        df = df.drop(columns=['A2', 'A3', 'A4', 'A5', 'A15'])
        df = df.rename(columns={'A6':'Avg',
                                                                'A7':'HR',
                                                                'A8':'R',
                                                                'A9':'RBI',
                                                                'A10':'SB',
                                                                'A11':'BB_Batting',
                                                                'A12':'SO_Batting'})
        df = df.rename(columns={'A6_extra':'IP',
                                                                'A7_extra':'BB_Pitching',
                                                                'A8_extra':'SO_Pitching',
                                                                'A9_extra':'W',
                                                                'A10_extra':'SV',
                                                                'A11_extra':'ERA',
                                                                'A12_extra':'WHIP'})

        df = adjust_pitcher_data(df)
        df.loc[(df['Role'] == 'Hitter') & (df['Player'] != 'Shohei Ohtani'), ['BB', 'SO', 'W', 'SV', 'ERA', 'WHIP']] = None
        df = df.rename(columns={'A13':'FPts',
                                                                'A14':'Avg FPts',
                                                                'A17':'Current Lineup Designation'})
        del df['A16']
        del df['SO']
        del df['BB']
        df['IP']= np.where(df['Role'] == 'Hitter', None, df['IP'])
        df['SO_Pitching']= np.where(df['Role'] == 'Hitter', None, df['SO_Pitching'])
        df['BB_Pitching'] = np.where(df['Role'] == 'Hitter', None, df['BB_Pitching'])
        return df

    for team, roster in team_rosters.items():
        team_rosters[team] = process_roster(roster)
        team_rosters[team]['Team'] = team
    
    all_rosters = pd.concat([team_roster for team_roster in team_rosters.values()], ignore_index=True)

    # Convert the DataFrame to an HTML table    
    # Return the HTML table as the response
    return func.HttpResponse(
        body=json.dumps({
            "all_rosters": all_rosters.to_dict(orient="records"),
            "standings": standings.to_dict(orient="records"),
            "pts_by_team_and_pos": pts_by_team_and_pos.to_dict(orient="records")
        }),
        mimetype="application/json",
        status_code=200
    )