# WCPredict
WCPredict
Using historical World Cup match data (1930-2022) to predict the 2026 FIFA World Cup.
About
This project builds a machine learning pipeline that analyzes 944 historical World Cup matches to predict outcomes for the 2026 tournament. It engineers features from raw match data, compares multiple classification models, and simulates the entire tournament from group stage through the final.
What it does

Cleans and processes match-level data spanning 22 World Cups
Engineers 22 features per match including rolling win rates, goal averages, head-to-head records, and host advantage
Compares Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
Simulates all 12 group stages with standings and advancement
Runs a full knockout bracket from Round of 32 to the Final

Key results
MetricValueBest modelLogistic RegressionCV accuracy (5-fold)59.2%Training matches926Features22Predicted championNetherlandsPredicted runner-upFrance
Data
Two datasets are used:

world_cups_1930_to_2022.txt - Match-level data with 944 rows. Each row is a single game with home/away teams, scores, result (H/A/D), round, and a neutral venue flag.
worldcups.csv - Tournament-level summaries (1930-2018) with host country, winner, attendance, and total goals. Used to derive host advantage features.

Features
The model uses 22 engineered features, computed using only data from prior tournaments to prevent data leakage:

Team strength: rolling win rate, draw rate, average goals scored/conceded, goal differential (last 5 tournaments)
Matchup context: head-to-head win rate and number of prior meetings
Relative features: differences in win rate, goal differential, and scoring between the two teams
Situational: whether either team is the host country, and whether the match is a knockout game

How to run
Requirements
pip install pandas numpy scikit-learn xgboost
Run the pipeline
python wc_predict.py
This prints the full output: model comparison, group stage results, knockout bracket, and the predicted champion.
Limitations

World Cup data only - the model doesn't use friendlies, qualifiers, or FIFA rankings, which would add valuable signal
Draw prediction is weak - the model predicts draws with only 4% recall since they're inherently hard to call
Group assignments are approximate - the actual 2026 draw may differ from what's used here
First 48-team World Cup - the bracket structure is simplified since there's no historical precedent for this format
Limited history for some teams - newer World Cup participants (e.g., Albania, Venezuela) have very few historical matches to learn from

Future improvements

Add FIFA rankings as a feature
Include international friendly and qualifier match data
Add visualizations (confusion matrix, feature importance, bracket diagram)
Try an Elo rating system as an alternative approach
Model goal counts (Poisson regression) instead of just win/loss/draw

Built with

Python, pandas, NumPy
scikit-learn, XGBoost
