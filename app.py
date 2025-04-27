import pandas as pd
import streamlit as st
import datetime
from rapidfuzz import process

# --- Helper Functions ---

def implied_probability(odds):
    odds = pd.to_numeric(odds, errors='coerce')
    if pd.isna(odds):
        return 0
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def highlight_over_odds(row):
    try:
        odds = pd.to_numeric(row['Over Odds'], errors='coerce')
        return ['background-color: #fff9c4' if odds <= -200 else '' for _ in row]
    except Exception:
        return ['' for _ in row]

def match_player(name, candidates):
    match, score, _ = process.extractOne(name, candidates, score_cutoff=70)
    return match

def slugify(name):
    return name.lower().replace(" ", "-").replace("é", "e").replace(".", "").replace("'", "")

def parse_odds_file_from_streamlit(uploaded_txt_file):
    lines = uploaded_txt_file.read().decode('utf-8').splitlines()

    current_player = None
    current_matchup = None
    data = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == '+' or any(book in line.lower() for book in ["fanduel", "draftkings", "caesars", "bet365", "betmgm", "riverscasino"]):
            i += 1
            continue

        if line.lower().startswith('u0.5') or line.lower().startswith('u1.5'):
            i += 1
            continue

        if line.lower().startswith('o0.5'):
            over_odds = ''
            if (i + 1) < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('-') or next_line.startswith('+') or next_line.lower() == 'even':
                    over_odds = next_line
                    i += 1
            if current_player and current_matchup:
                data.append({
                    "Player": current_player,
                    "Matchup": current_matchup,
                    "Line Type": line,
                    "Over Odds": over_odds
                })
                current_player = None
                current_matchup = None
            i += 1
            continue

        # Assume player name
        current_player = line
        if (i + 1) < len(lines):
            next_line = lines[i + 1].strip()
            if 'vs' in next_line or '@' in next_line:
                current_matchup = next_line
                i += 1

        i += 1

    df = pd.DataFrame(data)
    return df

# --- Main App ---

def main():
    st.set_page_config(layout="wide")
    st.title('⚾ MLB Hit Prop Model - Daily Recommendations')

    st.sidebar.header("📂 Upload Your Data Files")
    player_stats_file = st.sidebar.file_uploader('Upload Player Stats CSV', type='csv')
    betting_odds_file = st.sidebar.file_uploader('Upload Betting Odds CSV', type='csv')
    recent_hits_file = st.sidebar.file_uploader('Upload Recent Hits CSV', type='csv')
    pitcher_stats_file = st.sidebar.file_uploader('Upload Pitcher Stats CSV', type='csv')
    results_file = st.sidebar.file_uploader('Upload Game Results CSV', type='csv')

    with st.sidebar.expander("🧹 Parse Raw Odds Text File"):
        uploaded_odds_txt = st.file_uploader("Upload Raw Odds TXT", type=['txt'])

        if uploaded_odds_txt:
            parsed_odds_df = parse_odds_file_from_streamlit(uploaded_odds_txt)

            if not parsed_odds_df.empty:
                st.success(f"✅ Parsed {len(parsed_odds_df)} odds entries!")
                st.dataframe(parsed_odds_df, use_container_width=True)

                csv = parsed_odds_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Parsed Odds CSV",
                    data=csv,
                    file_name="parsed_betting_odds.csv",
                    mime='text/csv'
                )
            else:
                st.error("❌ Failed to parse any valid odds data. Check formatting.")

    if player_stats_file and betting_odds_file:
        player_stats = pd.read_csv(player_stats_file)
        betting_odds = pd.read_csv(betting_odds_file)

        player_stats.columns = player_stats.columns.str.strip()
        betting_odds.columns = betting_odds.columns.str.strip()

        if 'OverOdds' in betting_odds.columns:
            betting_odds.rename(columns={'OverOdds': 'Over Odds'}, inplace=True)

        if 'Player Name' in player_stats.columns:
            player_stats.rename(columns={'Player Name': 'Player'}, inplace=True)
        if 'Player Name' in betting_odds.columns:
            betting_odds.rename(columns={'Player Name': 'Player'}, inplace=True)

        if 'last_name, first_name' in player_stats.columns:
            split_names = player_stats['last_name, first_name'].str.split(', ', expand=True)
            player_stats['Player'] = split_names[1] + ' ' + split_names[0]

        player_stats.rename(columns={
            'hard_hit_percent': 'hard_hit_rate',
            'barrel_batted_rate': 'barrel_rate',
            'sweet_spot_percent': 'sweet_spot_rate'
        }, inplace=True)

        if 'Player' in player_stats.columns and 'Player' in betting_odds.columns:
            df = pd.merge(betting_odds, player_stats, on='Player', how='left')

            required_stats = ['batting_avg', 'xba', 'on_base_percent', 'babip', 'hard_hit_rate', 'barrel_rate', 'sweet_spot_rate']
            for stat in required_stats:
                if stat not in df.columns:
                    df[stat] = 0
                else:
                    mean = df[stat].mean()
                    std = df[stat].std()
                    if std != 0:
                        df[stat] = (df[stat] - mean) / std
                    else:
                        df[stat] = 0

            # --- Recent Hits Adjustment ---
            if recent_hits_file:
                recent_hits = pd.read_csv(recent_hits_file)
                recent_hits.columns = recent_hits.columns.str.strip()
                if 'Player' in recent_hits.columns and 'Last7_Hits' in recent_hits.columns:
                    df = pd.merge(df, recent_hits, on='Player', how='left')
                    df['Last7_Hits'] = df['Last7_Hits'].fillna(0)
                    df['Model_Hot_Streak_Boost'] = df['Last7_Hits'] / 7 * 0.05
                else:
                    st.warning("Recent Hits file must contain 'Player' and 'Last7_Hits' columns.")
                    df['Model_Hot_Streak_Boost'] = 0
            else:
                df['Model_Hot_Streak_Boost'] = 0

            # --- Build Base Model Hit Probability ---
            df['Model_Hit_Prob'] = (
                0.25 * df['batting_avg'] +
                0.25 * df['xba'] +
                0.15 * df['on_base_percent'] +
                0.10 * df['babip'] +
                0.10 * df['hard_hit_rate'] +
                0.075 * df['barrel_rate'] +
                0.075 * df['sweet_spot_rate'] +
                df['Model_Hot_Streak_Boost']
            )

            # --- Pitcher Adjustment ---
            if pitcher_stats_file:
                pitcher_stats = pd.read_csv(pitcher_stats_file)
                pitcher_stats.columns = pitcher_stats.columns.str.strip()

                if 'Matchup' in df.columns:
                    df['Opponent_Team'] = df['Matchup'].apply(lambda x: x.split('vs')[-1].strip() if 'vs' in x else '')
                    df = pd.merge(df, pitcher_stats, left_on='Opponent_Team', right_on='Team', how='left', suffixes=('', '_Pitcher'))

                    df['ERA'] = df['ERA'].fillna(4.50)
                    df['WHIP'] = df['WHIP'].fillna(1.30)
                    df['Hits_Allowed'] = df['Hits_Allowed'].fillna(50)

                    df['Pitcher_Adjustment'] = (df['WHIP'] - 1.30) * 0.05
                    df['Model_Hit_Prob'] = (df['Model_Hit_Prob'] + df['Pitcher_Adjustment']).clip(0, 1)

            # --- Final Calculations ---
            df['Implied_Prob'] = df['Over Odds'].apply(implied_probability)
            df['Edge_%'] = (df['Model_Hit_Prob'] - df['Implied_Prob']) * 100
            df['Confidence_%'] = ((0.7 * df['Model_Hit_Prob'] + 0.3 * (df['Edge_%'] / 100)) * 100).clip(0, 100)
            df['Recommended_Bet'] = df['Edge_%'] > 5
            df['Recommended_Bet'] = df['Recommended_Bet'].apply(lambda x: '✅' if x else '')

            output = df[['Player', 'Matchup', 'Over Odds', 'Model_Hit_Prob', 'Implied_Prob', 'Edge_%', 'Confidence_%', 'Recommended_Bet']]
            output = output.sort_values(by='Edge_%', ascending=False)
            output.insert(0, 'Rank', range(1, len(output) + 1))
            output['Results'] = ''

            # --- Process Results (Optional) ---
            if results_file:
                results_df = pd.read_csv(results_file)
                results_df.columns = results_df.columns.str.strip()
                if 'Player' in results_df.columns and 'H' in results_df.columns:
                    hit_results = dict(zip(results_df['Player'], results_df['H']))
                    output['Results'] = output['Player'].apply(lambda x: '✅' if hit_results.get(match_player(x, hit_results.keys()), 0) >= 1 else '❌')
                else:
                    st.warning("Results file must contain 'Player' and 'H' columns.")

            # --- Display Output ---
            st.subheader('📈 Recommended Bets')
            if st.checkbox("Show FanGraphs Game Log Links"):
                output['Game Log URL'] = output['Player'].apply(lambda name: f"https://www.fangraphs.com/players/{slugify(name)}/00000/game-log?season=2025&type=1")

            edited_output = st.data_editor(
                output,
                use_container_width=True,
                column_config={
                    'Results': st.column_config.SelectboxColumn(
                        label='Results',
                        options=['', '✅', '❌'],
                        required=False
                    )
                },
                disabled=['Rank', 'Player', 'Matchup', 'Over Odds', 'Model_Hit_Prob', 'Implied_Prob', 'Edge_%', 'Confidence_%', 'Recommended_Bet']
            )

            if results_file:
                wins = (edited_output['Results'] == '✅').sum()
                losses = (edited_output['Results'] == '❌').sum()
                total = wins + losses
                win_rate = (wins / total * 100) if total > 0 else 0
                st.write(f"✅ Wins: {wins} | ❌ Losses: {losses} | 🏆 Win %: {win_rate:.1f}%")

            styled_output = edited_output.style.apply(highlight_over_odds, axis=1)
            st.dataframe(styled_output, use_container_width=True)

            today = datetime.date.today().strftime("%Y-%m-%d")
            csv = edited_output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Recommendations as CSV',
                data=csv,
                file_name=f'recommended_bets_{today}.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
