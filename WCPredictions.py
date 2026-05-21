"""
World Cup 2026 Predictions Pipeline
Uses historical World Cup match data (1930-2022) to predict 2026 outcomes.

Denise Anciola
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

matches = pd.read_csv('world_cups_1980_to_2022.csv')
worldcups = pd.read_csv('worldcups.csv')

# Parse dates
matches['date'] = pd.to_datetime(matches['date'], format='%d/%m/%Y')
matches = matches.sort_values('date').reset_index(drop=True)

# Convert scores to int
matches['home_score'] = matches['home_score'].astype(int)
matches['away_score'] = matches['away_score'].astype(int)

# Create host lookup from worldcups.csv (only goes to 2018, add 2022 manually)
host_map = dict(zip(worldcups['year'], worldcups['host']))
host_map[2022] = 'Qatar'

# Handle multi-host entries
def get_host_countries(host_str):
    """Split 'Japan, South Korea' into list."""
    return [h.strip() for h in str(host_str).split(',')]

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Total matches: {len(matches)}")
print(f"Tournaments: {matches['season'].nunique()} ({matches['season'].min()}-{matches['season'].max()})")
print(f"Unique teams: {len(set(matches['home']) | set(matches['away']))}")
print(f"Outcome distribution: {matches['result'].value_counts().to_dict()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_team_history(df):
    """
    For each match, compute rolling historical features for both teams.
    Only uses data from PRIOR tournaments to avoid leakage.
    """
    
    # Collect all matches into a team-level history
    records = []
    for _, row in df.iterrows():
        # Home team record
        records.append({
            'team': row['home'],
            'opponent': row['away'],
            'season': row['season'],
            'date': row['date'],
            'goals_for': row['home_score'],
            'goals_against': row['away_score'],
            'result': 1 if row['result'] == 'H' else (0 if row['result'] == 'D' else -1),
            'is_home': 1
        })
        # Away team record
        records.append({
            'team': row['away'],
            'opponent': row['home'],
            'season': row['season'],
            'date': row['date'],
            'goals_for': row['away_score'],
            'goals_against': row['home_score'],
            'result': 1 if row['result'] == 'A' else (0 if row['result'] == 'D' else -1),
            'is_home': 0
        })
    
    return pd.DataFrame(records)


def compute_team_stats(team_history, team, before_season, n_recent_tournaments=5):
    """Compute rolling stats for a team using only data before a given season."""
    
    hist = team_history[
        (team_history['team'] == team) & 
        (team_history['season'] < before_season)
    ]
    
    if len(hist) == 0:
        return {
            'win_rate': 0.33,
            'draw_rate': 0.33,
            'avg_goals_for': 1.0,
            'avg_goals_against': 1.0,
            'goal_diff': 0.0,
            'matches_played': 0,
            'tournaments_played': 0,
        }
    
    # Recent tournaments only (more predictive than all-time)
    recent_seasons = sorted(hist['season'].unique())[-n_recent_tournaments:]
    recent = hist[hist['season'].isin(recent_seasons)]
    
    if len(recent) == 0:
        recent = hist  # fallback
    
    wins = (recent['result'] == 1).sum()
    draws = (recent['result'] == 0).sum()
    total = len(recent)
    
    return {
        'win_rate': wins / total,
        'draw_rate': draws / total,
        'avg_goals_for': recent['goals_for'].mean(),
        'avg_goals_against': recent['goals_against'].mean(),
        'goal_diff': (recent['goals_for'] - recent['goals_against']).mean(),
        'matches_played': total,
        'tournaments_played': recent['season'].nunique(),
    }


def compute_h2h(team_history, team1, team2, before_season):
    """Head-to-head record between two teams before a given season."""
    h2h = team_history[
        (team_history['team'] == team1) &
        (team_history['opponent'] == team2) &
        (team_history['season'] < before_season)
    ]
    if len(h2h) == 0:
        return {'h2h_win_rate': 0.5, 'h2h_matches': 0}
    
    wins = (h2h['result'] == 1).sum()
    return {
        'h2h_win_rate': wins / len(h2h),
        'h2h_matches': len(h2h)
    }


print("\nBuilding team history...")
team_history = build_team_history(matches)

print("Engineering features for each match...")
feature_rows = []

for idx, row in matches.iterrows():
    season = row['season']
    home = row['home']
    away = row['away']
    
    # Get stats for each team
    home_stats = compute_team_stats(team_history, home, season)
    away_stats = compute_team_stats(team_history, away, season)
    
    # Head-to-head
    h2h = compute_h2h(team_history, home, away, season)
    
    # Host advantage
    hosts = get_host_countries(host_map.get(season, ''))
    home_is_host = 1 if home in hosts else 0
    away_is_host = 1 if away in hosts else 0
    
    # Stage encoding (knockout vs group)
    is_knockout = 0 if 'group' in row['round'].lower() or 'Group' in row['round'] else 1
    
    feature_rows.append({
        # Identifiers
        'season': season,
        'home': home,
        'away': away,
        'result': row['result'],
        
        # Home team features
        'home_win_rate': home_stats['win_rate'],
        'home_draw_rate': home_stats['draw_rate'],
        'home_avg_gf': home_stats['avg_goals_for'],
        'home_avg_ga': home_stats['avg_goals_against'],
        'home_goal_diff': home_stats['goal_diff'],
        'home_matches': home_stats['matches_played'],
        'home_tournaments': home_stats['tournaments_played'],
        
        # Away team features
        'away_win_rate': away_stats['win_rate'],
        'away_draw_rate': away_stats['draw_rate'],
        'away_avg_gf': away_stats['avg_goals_for'],
        'away_avg_ga': away_stats['avg_goals_against'],
        'away_goal_diff': away_stats['goal_diff'],
        'away_matches': away_stats['matches_played'],
        'away_tournaments': away_stats['tournaments_played'],
        
        # Relative features
        'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
        'goal_diff_diff': home_stats['goal_diff'] - away_stats['goal_diff'],
        'avg_gf_diff': home_stats['avg_goals_for'] - away_stats['avg_goals_for'],
        
        # Head-to-head
        'h2h_win_rate': h2h['h2h_win_rate'],
        'h2h_matches': h2h['h2h_matches'],
        
        # Context features
        'home_is_host': home_is_host,
        'away_is_host': away_is_host,
        'is_knockout': is_knockout,
    })

features_df = pd.DataFrame(feature_rows)

# Drop first tournament (1930) -- no prior history to use
features_df = features_df[features_df['season'] > 1930].copy()

print(f"Feature matrix: {features_df.shape[0]} matches x {len([c for c in features_df.columns if c not in ['season','home','away','result']])} features")

# ============================================================
# 3. MODEL TRAINING
# ============================================================

feature_cols = [c for c in features_df.columns if c not in ['season', 'home', 'away', 'result']]
X = features_df[feature_cols].values
y_raw = features_df['result'].values

# Encode target: H=0, D=1, A=2
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nTarget classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Compare models
print("\n" + "=" * 60)
print("MODEL COMPARISON (5-Fold Stratified CV)")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, 
                              use_label_encoder=False, eval_metric='mlogloss', random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_model_name = None

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    print(f"  {name:25s}  Accuracy: {mean_score:.3f} (+/- {scores.std():.3f})")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\nBest model: {best_model_name} ({best_score:.3f})")

# Train final model on all data
best_model = models[best_model_name]
best_model.fit(X, y)

# Classification report on training data (for reference)
y_pred = best_model.predict(X)
print(f"\n{'='*60}")
print(f"CLASSIFICATION REPORT ({best_model_name} on full training set)")
print("="*60)
print(classification_report(y_raw, le.inverse_transform(y_pred), 
                            target_names=['Away Win', 'Draw', 'Home Win']))

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance = pd.Series(best_model.feature_importances_, index=feature_cols)
    print("\nTop 10 Feature Importances:")
    for feat, imp in importance.sort_values(ascending=False).head(10).items():
        print(f"  {feat:25s}  {imp:.4f}")

# ============================================================
# 4. PREDICT 2026 WORLD CUP
# ============================================================

print(f"\n{'='*60}")
print("2026 WORLD CUP PREDICTIONS")
print("="*60)

# 2026 group stage matchups (confirmed groups)
# Host: USA, Canada, Mexico
groups_2026 = {
    'A': ['USA', 'Panama', 'Bolivia', 'New Zealand'],         # not yet fully confirmed
    'B': ['Argentina', 'Peru', 'Morocco', 'Denmark'],
    'C': ['Mexico', 'Ecuador', 'Senegal', 'Albania'],         # not yet fully confirmed  
    'D': ['France', 'Colombia', 'Saudi Arabia', 'Australia'],
    'E': ['Brazil', 'Japan', 'Turkey', 'Paraguay'],
    'F': ['England', 'Uruguay', 'Chile', 'Tunisia'],
    'G': ['Spain', 'Nigeria', 'Iran', 'Canada'],
    'H': ['Portugal', 'South Korea', 'Serbia', 'Costa Rica'],
    'I': ['Netherlands', 'Cameroon', 'Switzerland', 'Scotland'],
    'J': ['Italy', 'Croatia', 'Ukraine', 'Honduras'],
    'K': ['Germany', 'Ghana', 'Qatar', 'Egypt'],
    'L': ['Belgium', 'Poland', 'Venezuela', 'Wales'],
}

# Teams that might not be in our historical data -- we'll handle gracefully
host_countries_2026 = ['USA', 'Canada', 'Mexico']

def predict_match(model, team_history, home, away, season=2026, le=le, feature_cols=feature_cols):
    """Predict a single match outcome with probabilities."""
    
    home_stats = compute_team_stats(team_history, home, season)
    away_stats = compute_team_stats(team_history, away, season)
    h2h = compute_h2h(team_history, home, away, season)
    
    home_is_host = 1 if home in host_countries_2026 else 0
    away_is_host = 1 if away in host_countries_2026 else 0
    
    feat_dict = {
        'home_win_rate': home_stats['win_rate'],
        'home_draw_rate': home_stats['draw_rate'],
        'home_avg_gf': home_stats['avg_goals_for'],
        'home_avg_ga': home_stats['avg_goals_against'],
        'home_goal_diff': home_stats['goal_diff'],
        'home_matches': home_stats['matches_played'],
        'home_tournaments': home_stats['tournaments_played'],
        'away_win_rate': away_stats['win_rate'],
        'away_draw_rate': away_stats['draw_rate'],
        'away_avg_gf': away_stats['avg_goals_for'],
        'away_avg_ga': away_stats['avg_goals_against'],
        'away_goal_diff': away_stats['goal_diff'],
        'away_matches': away_stats['matches_played'],
        'away_tournaments': away_stats['tournaments_played'],
        'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
        'goal_diff_diff': home_stats['goal_diff'] - away_stats['goal_diff'],
        'avg_gf_diff': home_stats['avg_goals_for'] - away_stats['avg_goals_for'],
        'h2h_win_rate': h2h['h2h_win_rate'],
        'h2h_matches': h2h['h2h_matches'],
        'home_is_host': home_is_host,
        'away_is_host': away_is_host,
        'is_knockout': 0,  # default group stage
    }
    
    X_pred = np.array([[feat_dict[c] for c in feature_cols]])
    proba = model.predict_proba(X_pred)[0]
    pred_class = le.inverse_transform([model.predict(X_pred)[0]])[0]
    
    # Map probabilities to labels
    prob_dict = dict(zip(le.classes_, proba))
    
    return pred_class, prob_dict


# Simulate group stage
print("\nGROUP STAGE PREDICTIONS:")
print("-" * 60)

group_results = {}

for group_name, teams in groups_2026.items():
    print(f"\n  Group {group_name}: {', '.join(teams)}")
    
    # Points table
    points = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    
    # Round-robin within group
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            home = teams[i]
            away = teams[j]
            
            pred, probs = predict_match(best_model, team_history, home, away)
            
            h_prob = probs.get('H', 0)
            d_prob = probs.get('D', 0)
            a_prob = probs.get('A', 0)
            
            if pred == 'H':
                points[home] += 3
                gf[home] += 1; ga[away] += 1
            elif pred == 'A':
                points[away] += 3
                gf[away] += 1; ga[home] += 1
            else:
                points[home] += 1
                points[away] += 1
            
            print(f"    {home:20s} vs {away:20s} -> {pred}  (H:{h_prob:.0%} D:{d_prob:.0%} A:{a_prob:.0%})")
    
    # Sort by points, then goal difference
    standings = sorted(teams, key=lambda t: (points[t], gf[t] - ga[t]), reverse=True)
    
    print(f"    Standings: ", end="")
    for rank, t in enumerate(standings, 1):
        gd = gf[t] - ga[t]
        print(f"{rank}. {t} ({points[t]}pts, GD:{gd:+d}) ", end="")
    print()
    
    # Top 2 advance (simplified -- actual 2026 has top 2 + best 3rds from 48-team format)
    group_results[group_name] = {
        'standings': standings,
        'points': points,
        'gf': gf,
        'ga': ga,
        'qualified': standings[:2]  # Top 2 advance
    }

# Summary: who advances
print(f"\n{'='*60}")
print("TEAMS ADVANCING FROM GROUP STAGE (Top 2 per group)")
print("="*60)
advancing = []
for g in sorted(group_results.keys()):
    q = group_results[g]['qualified']
    advancing.extend(q)
    print(f"  Group {g}: {q[0]}, {q[1]}")

print(f"\n  Total advancing: {len(advancing)} teams")

# ============================================================
# 5. SIMULATE KNOCKOUT ROUNDS
# ============================================================

print(f"\n{'='*60}")
print("KNOCKOUT STAGE SIMULATION")
print("="*60)

# Simplified bracket: 1A vs 2B, 1B vs 2A, etc.
def simulate_knockout_round(matchups, team_history, model, round_name):
    """Simulate a knockout round. No draws -- pick the team with higher win probability."""
    print(f"\n  {round_name}:")
    winners = []
    for home, away in matchups:
        pred, probs = predict_match(model, team_history, home, away)
        
        h_prob = probs.get('H', 0)
        a_prob = probs.get('A', 0)
        
        # In knockout, no draws -- pick likely winner
        if h_prob >= a_prob:
            winner = home
            win_prob = h_prob + probs.get('D', 0) * 0.5  # split draw prob
        else:
            winner = away
            win_prob = a_prob + probs.get('D', 0) * 0.5
        
        loser = away if winner == home else home
        print(f"    {home:20s} vs {away:20s} -> Winner: {winner} ({win_prob:.0%})")
        winners.append(winner)
    
    return winners

# 2026 format: 12 groups, top 2 per group = 24 auto-qualifiers
# + 8 best 3rd-place teams = 32 total in knockout
# Determine best 3rd-place teams
third_place = []
for g in sorted(group_results.keys()):
    t = group_results[g]['standings'][2]  # 3rd place team
    pts = group_results[g]['points'][t]
    gd = group_results[g]['gf'][t] - group_results[g]['ga'][t]
    gf_val = group_results[g]['gf'][t]
    third_place.append({'team': t, 'group': g, 'points': pts, 'gd': gd, 'gf': gf_val})

third_place_df = pd.DataFrame(third_place)
third_place_df = third_place_df.sort_values(['points', 'gd', 'gf'], ascending=False)
best_thirds = third_place_df.head(8)['team'].tolist()

print(f"\n  Best 3rd-place qualifiers: {', '.join(best_thirds)}")

# All 32 knockout teams
knockout_teams = []
for g in sorted(group_results.keys()):
    knockout_teams.extend(group_results[g]['qualified'])  # top 2
knockout_teams.extend(best_thirds)

print(f"  Total knockout teams: {len(knockout_teams)}")

# Build Round of 32 bracket (simplified: pair groups A-B, C-D, etc., 
# with 3rd-place teams slotted into the bracket)
group_keys = sorted(group_results.keys())

# Pair 1st-place vs 2nd-place across adjacent groups, plus 3rd place matchups
r32_matchups = []
for i in range(0, len(group_keys), 2):
    g1 = group_keys[i]
    g2 = group_keys[i+1]
    r32_matchups.append((group_results[g1]['qualified'][0], group_results[g2]['qualified'][1]))
    r32_matchups.append((group_results[g2]['qualified'][0], group_results[g1]['qualified'][1]))

# Add 3rd place matchups (pair them up)
for i in range(0, len(best_thirds), 2):
    # Match best 3rds against each other or against remaining group winners
    r32_matchups.append((best_thirds[i], best_thirds[i+1]))

r32_winners = simulate_knockout_round(r32_matchups, team_history, best_model, "Round of 32")

# Round of 16 (16 matches -> 8 winners... wait, we have 16 R32 matches -> 16 winners)
r16_matchups = [(r32_winners[i], r32_winners[i+1]) for i in range(0, len(r32_winners), 2)]
r16_winners = simulate_knockout_round(r16_matchups, team_history, best_model, "Round of 16")

# Quarterfinals
qf_matchups = [(r16_winners[i], r16_winners[i+1]) for i in range(0, len(r16_winners), 2)]
qf_winners = simulate_knockout_round(qf_matchups, team_history, best_model, "Quarterfinals")

# Semifinals
sf_matchups = [(qf_winners[i], qf_winners[i+1]) for i in range(0, len(qf_winners), 2)]
sf_winners = simulate_knockout_round(sf_matchups, team_history, best_model, "Semifinals")

# Final
print(f"\n  {'='*40}")
final_pred, final_probs = predict_match(best_model, team_history, sf_winners[0], sf_winners[1])
h_prob = final_probs.get('H', 0)
a_prob = final_probs.get('A', 0)
if h_prob >= a_prob:
    champion = sf_winners[0]
    runner_up = sf_winners[1]
else:
    champion = sf_winners[1]
    runner_up = sf_winners[0]

print(f"  FINAL: {sf_winners[0]} vs {sf_winners[1]}")
print(f"  {'='*40}")
print(f"\n  🏆 PREDICTED 2026 WORLD CUP CHAMPION: {champion}")
print(f"  🥈 Runner-up: {runner_up}")

# Save results for visualization
results_summary = {
    'champion': champion,
    'runner_up': runner_up,
    'semifinalists': sf_winners,
    'quarterfinalists': qf_winners,
    'group_results': {g: group_results[g]['standings'] for g in group_results},
    'model': best_model_name,
    'cv_accuracy': best_score,
}
