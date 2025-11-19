standings = client.standings.league_standings(season="20242025")

df = pd.DataFrame(data['standings'])

# Flatten nested fields
df['team'] = df['teamName'].apply(lambda x: x.get('default'))
df['abbrev'] = df['teamAbbrev'].apply(lambda x: x.get('default'))
df['city'] = df['placeName'].apply(lambda x: x.get('default'))

# Select only important columns for now
df = df[['team', 'abbrev', 'city', 'conferenceName', 'divisionName',
         'gamesPlayed', 'wins', 'losses', 'otLosses', 'points', 'pointPctg',
         'goalFor', 'goalAgainst', 'goalDifferential', 'streakCode', 'streakCount']]