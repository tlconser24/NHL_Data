from nhlpy import NHLClient
import pandas as pd


# Default configuration
client = NHLClient()

# With debug logging
client = NHLClient(debug=True)

# All available configurations
client = NHLClient(
    debug=True,           # Enable debug logging
    timeout=30,           # Request timeout in seconds
    ssl_verify=True,      # SSL certificate verification
    follow_redirects=True # Follow HTTP redirects
)

standings = client.standings.league_standings(season="20242025")

standing_df = pd.DataFrame(standings['standings'])

# Flatten nested fields
standing_df['team'] = standing_df['teamName'].apply(lambda x: x.get('default'))
standing_df['abbrev'] = standing_df['teamAbbrev'].apply(lambda x: x.get('default'))
standing_df['city'] = standing_df['placeName'].apply(lambda x: x.get('default'))

# Select only important columns for now
nhl_standing_df = standing_df[['team', 'abbrev', 'city', 'conferenceName', 'divisionName',
         'gamesPlayed', 'wins', 'losses', 'otLosses', 'points', 'pointPctg',
         'goalFor', 'goalAgainst', 'goalDifferential', 'streakCode', 'streakCount']]

#print(nhl_standing_df)
#print(nhl_standing_df.dtypes)
#print(nhl_standing_df.shape)
#print(nhl_standing_df.columns)

import os

output_path = "data/team/team_standings_2025.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

nhl_standing_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
