# Used libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar


def preprocess_file(input_path, output_path):

    data_df = pd.read_csv(input_path)

    # For Airline = NaN, fill the value from flight_no
    def fill_missing_airline(row):
        if not isinstance(row['Airline'], str) and np.isnan(row['Airline']):
            row['Airline'] = row['flight_no'][:2]
        return row

    data_df = data_df.apply(fill_missing_airline, axis=1)

    # Get flight datetime-related columns
    data_df['flight_date_dt'] = data_df['flight_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data_df['flight_dt'] = data_df.apply(lambda x: x['flight_date_dt'] + timedelta(hours=x['std_hour']), axis=1)
    data_df['flight_year'] = data_df['flight_dt'].apply(lambda x: x.year)
    data_df['flight_month'] = data_df['flight_dt'].apply(lambda x: x.month)
    data_df['flight_day'] = data_df['flight_dt'].apply(lambda x: x.day)
    data_df['flight_2_day_bin'] = data_df['flight_dt'].apply(lambda x: x.day // 2)
    data_df['flight_4_day_bin'] = data_df['flight_dt'].apply(lambda x: x.day // 4)
    data_df['flight_2_hour_bin'] = data_df['std_hour'].apply(lambda x: x // 2)
    data_df['flight_4_hour_bin'] = data_df['std_hour'].apply(lambda x: x // 4)

    # Before processing delay_time statistics, remove cancalled entries first
    non_cancel_df = data_df[~(data_df['delay_time'] == "Cancelled")].copy()

    # Translate delay_time to float
    non_cancel_df['delay_time'] = non_cancel_df['delay_time'].apply(lambda x: float(x))

    # Note: groupby preserves the order of rows within each group
    non_cancel_df = non_cancel_df.sort_values(['Departure', 'Arrival', 'Airline', 'flight_year', 'flight_month', 'flight_day', 'std_hour'],ascending=False)

    # Aggregate per different perspectives by different bins
    perspectives = {'dep': ['Departure'],
                    'arr': ['Arrival'],
                    'air': ['Airline'],
                    'arr_air': ['Arrival', 'Airline']}

    time_bins = {'hr': ['flight_year', 'flight_month', 'flight_day', 'std_hour'],
                 '2_hr': ['flight_year', 'flight_month', 'flight_day', 'flight_2_hour_bin'],
                 '4_hr': ['flight_year', 'flight_month', 'flight_day', 'flight_4_hour_bin'],
                 'day': ['flight_year', 'flight_month', 'flight_day'],
                 '2_day': ['flight_year', 'flight_month', 'flight_2_day_bin'],
                 '4_day': ['flight_year', 'flight_month', 'flight_4_day_bin'],
                 'wk': ['flight_year', 'Week']}

    # Creation of different stat dfs
    perspective_time_dfs = {}
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key
            perspective_time = p + t
            perspective_time_dfs[pt_key] = non_cancel_df.groupby(perspective_time).mean()['delay_time'].reset_index()

    # Helper functions for getting timestamp values by different bins
    def get_hr_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_day']
        std_hour = int(row['std_hour'])
        return flight_year * (10 ** 6) + flight_month * (10 ** 4) + flight_day * (10 ** 2) + std_hour

    def get_2_hr_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_day']
        flight_2_hour_bin = row['flight_2_hour_bin']
        return flight_year * (10 ** 6) + flight_month * (10 ** 4) + flight_day * (10 ** 2) + flight_2_hour_bin

    def get_4_hr_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_day']
        flight_4_hour_bin = row['flight_4_hour_bin']
        return flight_year * (10 ** 6) + flight_month * (10 ** 4) + flight_day * (10 ** 2) + flight_4_hour_bin

    def get_day_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_day']
        return flight_year * (10 ** 4) + flight_month * (10 ** 2) + flight_day

    def get_2_day_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_2_day_bin']
        return flight_year * (10 ** 4) + flight_month * (10 ** 2) + flight_day

    def get_4_day_ts_val(row):
        flight_year = row['flight_year']
        flight_month = row['flight_month']
        flight_day = row['flight_4_day_bin']
        return flight_year * (10 ** 4) + flight_month * (10 ** 2) + flight_day

    def get_wk_ts_val(row):
        flight_year = row['flight_year']
        flight_wk = row['Week']
        return flight_year * (10 ** 2) + flight_wk

    # For each stat df, create a time series key such that it is easier to select "most recent" value in later stage
    # This key will not become a feature in model training

    for pt_key in perspective_time_dfs:
        if pt_key.endswith('_2_hr'):
            perspective_time_dfs[pt_key]['flight_2_hour_ts'] = perspective_time_dfs[pt_key].apply(get_2_hr_ts_val, axis=1)
        elif pt_key.endswith('_4_hr'):
            perspective_time_dfs[pt_key]['flight_4_hour_ts'] = perspective_time_dfs[pt_key].apply(get_4_hr_ts_val, axis=1)
        elif pt_key.endswith('_2_day'):
            perspective_time_dfs[pt_key]['flight_2_day_ts'] = perspective_time_dfs[pt_key].apply(get_2_day_ts_val, axis=1)
        elif pt_key.endswith('_4_day'):
            perspective_time_dfs[pt_key]['flight_4_day_ts'] = perspective_time_dfs[pt_key].apply(get_4_day_ts_val, axis=1)
        elif pt_key.endswith('_day'):
            perspective_time_dfs[pt_key]['flight_day_ts'] = perspective_time_dfs[pt_key].apply(get_day_ts_val, axis=1)
        elif pt_key.endswith('_wk'):
            perspective_time_dfs[pt_key]['flight_wk_ts'] = perspective_time_dfs[pt_key].apply(get_wk_ts_val, axis=1)
        elif pt_key.endswith('_hr'):
            perspective_time_dfs[pt_key]['flight_ts'] = perspective_time_dfs[pt_key].apply(get_hr_ts_val, axis=1)

        # Rename column for join operation
        perspective_time_dfs[pt_key].rename(columns={'delay_time': '_'.join([pt_key, 'delay_time'])}, inplace=True)

    delay_time_dfs = perspective_time_dfs.copy()

    # Do the same, but for having delay only times
    # Before processing delay_time statistics, remove cancalled entries first
    non_cancel_df = data_df[~(data_df['delay_time'] == "Cancelled")].copy()

    # Translate delay_time to float
    non_cancel_df['delay_time'] = non_cancel_df['delay_time'].apply(lambda x: float(x) if float(x) >= 3.0 else -100)
    have_delay_df = non_cancel_df[non_cancel_df['delay_time'] >= 3.0]

    # Note: groupby preserves the order of rows within each group
    have_delay_df = have_delay_df.sort_values(['Departure', 'Arrival', 'Airline', 'flight_year', 'flight_month', 'flight_day', 'std_hour'],ascending=False)

    # Creation of different stat dfs
    perspective_time_dfs = {}
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key
            perspective_time = p + t
            perspective_time_dfs[pt_key] = have_delay_df.groupby(perspective_time).mean()['delay_time'].reset_index()

    # For each stat df, create a time series key such that it is easier to select "most recent" value in later stage
    # This key will not become a feature in model training
    for pt_key in perspective_time_dfs:
        if pt_key.endswith('_2_hr'):
            perspective_time_dfs[pt_key]['flight_2_hour_ts'] = perspective_time_dfs[pt_key].apply(get_2_hr_ts_val, axis=1)
        elif pt_key.endswith('_4_hr'):
            perspective_time_dfs[pt_key]['flight_4_hour_ts'] = perspective_time_dfs[pt_key].apply(get_4_hr_ts_val, axis=1)
        elif pt_key.endswith('_2_day'):
            perspective_time_dfs[pt_key]['flight_2_day_ts'] = perspective_time_dfs[pt_key].apply(get_2_day_ts_val, axis=1)
        elif pt_key.endswith('_4_day'):
            perspective_time_dfs[pt_key]['flight_4_day_ts'] = perspective_time_dfs[pt_key].apply(get_4_day_ts_val, axis=1)
        elif pt_key.endswith('_day'):
            perspective_time_dfs[pt_key]['flight_day_ts'] = perspective_time_dfs[pt_key].apply(get_day_ts_val, axis=1)
        elif pt_key.endswith('_wk'):
            perspective_time_dfs[pt_key]['flight_wk_ts'] = perspective_time_dfs[pt_key].apply(get_wk_ts_val, axis=1)
        elif pt_key.endswith('_hr'):
            perspective_time_dfs[pt_key]['flight_ts'] = perspective_time_dfs[pt_key].apply(get_hr_ts_val, axis=1)

        # Rename column for join operation
        perspective_time_dfs[pt_key].rename(columns={'delay_time': '_'.join([pt_key, 'delay_only_time'])}, inplace=True)

    delay_only_time_dfs = perspective_time_dfs.copy()

    # # Delay status statistics

    # Before processing delay_time statistics, remove cancalled entries first
    non_cancel_df = data_df[~(data_df['delay_time'] == "Cancelled")].copy()

    # Translate delay_time to float
    non_cancel_df['is_delayed'] = non_cancel_df['delay_time'].apply(lambda x: float(x) >= 3.0)

    # Note: groupby preserves the order of rows within each group
    non_cancel_df = non_cancel_df.sort_values(['Departure', 'Arrival', 'Airline', 'flight_year', 'flight_month', 'flight_day', 'std_hour'],ascending=False)

    # Creation of different stat dfs
    perspective_time_dfs = {}
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key
            perspective_time = p + t
            perspective_time_dfs[pt_key] = non_cancel_df.groupby(perspective_time).sum()['is_delayed'].reset_index()

    # For each stat df, create a time series key such that it is easier to select "most recent" value in later stage
    # This key will not become a feature in model training
    for pt_key in perspective_time_dfs:
        if pt_key.endswith('_2_hr'):
            perspective_time_dfs[pt_key]['flight_2_hour_ts'] = perspective_time_dfs[pt_key].apply(get_2_hr_ts_val, axis=1)
        elif pt_key.endswith('_4_hr'):
            perspective_time_dfs[pt_key]['flight_4_hour_ts'] = perspective_time_dfs[pt_key].apply(get_4_hr_ts_val, axis=1)
        elif pt_key.endswith('_2_day'):
            perspective_time_dfs[pt_key]['flight_2_day_ts'] = perspective_time_dfs[pt_key].apply(get_2_day_ts_val, axis=1)
        elif pt_key.endswith('_4_day'):
            perspective_time_dfs[pt_key]['flight_4_day_ts'] = perspective_time_dfs[pt_key].apply(get_4_day_ts_val, axis=1)
        elif pt_key.endswith('_day'):
            perspective_time_dfs[pt_key]['flight_day_ts'] = perspective_time_dfs[pt_key].apply(get_day_ts_val, axis=1)
        elif pt_key.endswith('_wk'):
            perspective_time_dfs[pt_key]['flight_wk_ts'] = perspective_time_dfs[pt_key].apply(get_wk_ts_val, axis=1)
        elif pt_key.endswith('_hr'):
            perspective_time_dfs[pt_key]['flight_ts'] = perspective_time_dfs[pt_key].apply(get_hr_ts_val, axis=1)

        # Rename column for join operation
        perspective_time_dfs[pt_key].rename(columns={'is_delayed': '_'.join([pt_key, 'delay_count'])}, inplace=True)

    delay_count_dfs = perspective_time_dfs.copy()

    # Before processing delay_time statistics, remove cancalled entries first
    cancel_df = data_df[(data_df['delay_time'] == "Cancelled")].copy()

    # Translate delay_time to float
    cancel_df['is_cancel'] = cancel_df['delay_time'].apply(lambda x: x == "Cancelled")

    # Note: groupby preserves the order of rows within each group
    cancel_df = cancel_df.sort_values(['Departure', 'Arrival', 'Airline', 'flight_year', 'flight_month', 'flight_day', 'std_hour'],ascending=False)

    # Creation of different stat dfs
    perspective_time_dfs = {}
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key
            perspective_time = p + t
            perspective_time_dfs[pt_key] = cancel_df.groupby(perspective_time).sum()['is_cancel'].reset_index()

    # For each stat df, create a time series key such that it is easier to select "most recent" value in later stage
    # This key will not become a feature in model training
    for pt_key in perspective_time_dfs:
        if pt_key.endswith('_2_hr'):
            perspective_time_dfs[pt_key]['flight_2_hour_ts'] = perspective_time_dfs[pt_key].apply(get_2_hr_ts_val, axis=1)
        elif pt_key.endswith('_4_hr'):
            perspective_time_dfs[pt_key]['flight_4_hour_ts'] = perspective_time_dfs[pt_key].apply(get_4_hr_ts_val, axis=1)
        elif pt_key.endswith('_2_day'):
            perspective_time_dfs[pt_key]['flight_2_day_ts'] = perspective_time_dfs[pt_key].apply(get_2_day_ts_val, axis=1)
        elif pt_key.endswith('_4_day'):
            perspective_time_dfs[pt_key]['flight_4_day_ts'] = perspective_time_dfs[pt_key].apply(get_4_day_ts_val, axis=1)
        elif pt_key.endswith('_day'):
            perspective_time_dfs[pt_key]['flight_day_ts'] = perspective_time_dfs[pt_key].apply(get_day_ts_val, axis=1)
        elif pt_key.endswith('_wk'):
            perspective_time_dfs[pt_key]['flight_wk_ts'] = perspective_time_dfs[pt_key].apply(get_wk_ts_val, axis=1)
        elif pt_key.endswith('_hr'):
            perspective_time_dfs[pt_key]['flight_ts'] = perspective_time_dfs[pt_key].apply(get_hr_ts_val, axis=1)

        # Rename column for join operation
        perspective_time_dfs[pt_key].rename(columns={'is_cancel': '_'.join([pt_key, 'cancel_count'])}, inplace=True)

    cancel_count_dfs = perspective_time_dfs.copy()

    feature_df = data_df.copy()

    # Use lagged time series value for below columns creation
    # Thus, the stat values should be available upon actual prediction
    def get_last_time_series_val(row):
        flight_dt = row['flight_dt'] - timedelta(hours=1)
        return int(flight_dt.strftime("%Y%m%d%H"))

    feature_df['flight_ts'] = feature_df.apply(get_last_time_series_val, axis=1)

    def get_last_day_time_series_val(row):
        flight_dt = row['flight_dt']
        flight_hour_bin = row['flight_2_hour_bin']
        flight_hour_bin -= 1
        if flight_hour_bin < 0:
            flight_dt = row['flight_dt'] - timedelta(days=1)
            flight_hour_bin = 11
        return int(flight_dt.strftime("%Y%m%d")) * 100 + flight_hour_bin

    feature_df['flight_2_hour_ts'] = feature_df.apply(get_last_day_time_series_val, axis=1)

    def get_last_day_time_series_val(row):
        flight_dt = row['flight_dt']
        flight_hour_bin = row['flight_4_hour_bin']
        flight_hour_bin -= 1
        if flight_hour_bin < 0:
            flight_dt = row['flight_dt'] - timedelta(days=1)
            flight_hour_bin = 5
        return int(flight_dt.strftime("%Y%m%d")) * 100 + flight_hour_bin

    feature_df['flight_4_hour_ts'] = feature_df.apply(get_last_day_time_series_val, axis=1)


    # Do the same for day/week
    def get_last_day_time_series_val(row):
        flight_dt = row['flight_dt'] - timedelta(days=1)
        return int(flight_dt.strftime("%Y%m%d"))

    feature_df['flight_day_ts'] = feature_df.apply(get_last_day_time_series_val, axis=1)

    def get_last_day_time_series_val(row):
        flight_dt = row['flight_dt']
        flight_day_bin = row['flight_2_day_bin']
        flight_day_bin -= 1
        if flight_day_bin < 0:
            flight_dt = row['flight_dt'] - timedelta(days=20)   # Approx. for getting last month value
            flight_day_bin = calendar.monthrange(flight_dt.year, flight_dt.month)[1] // 2
        return int(flight_dt.strftime("%Y%m")) * 100 + flight_day_bin

    feature_df['flight_2_day_ts'] = feature_df.apply(get_last_day_time_series_val, axis=1)

    def get_last_day_time_series_val(row):
        flight_dt = row['flight_dt']
        flight_day_bin = row['flight_4_day_bin']
        flight_day_bin -= 1
        if flight_day_bin < 0:
            flight_dt = row['flight_dt'] - timedelta(days=20)   # Approx. for getting last month value
            flight_day_bin = calendar.monthrange(flight_dt.year, flight_dt.month)[1] // 4
        return int(flight_dt.strftime("%Y%m")) * 100 + flight_day_bin

    feature_df['flight_4_day_ts'] = feature_df.apply(get_last_day_time_series_val, axis=1)

    def get_number_of_weeks_in_year(year):
        last_week = datetime(year, 12, 28)
        return last_week.isocalendar()[1]

    def get_last_wk_time_series_val(row):
        flight_year = row['flight_year']
        flight_week = row['Week'] - 1

        if flight_week < 1:
            # Shift to last year end
            flight_year -= 1
            flight_week = get_number_of_weeks_in_year(flight_year)

        return flight_year * (10 ** 2) + flight_week

    feature_df['flight_wk_ts'] = feature_df.apply(get_last_wk_time_series_val, axis=1)

    merged_feature_df = feature_df.copy()

    # Merge generated stat to original dataset
    # The merge operation is based on last hr/day/wk from current row's datetime, thus it is assumed those statistics can be calculated for new predictions

    # Delay time
    perspective_time_dfs = delay_time_dfs
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key

            if pt_key.endswith('_2_hr'):
                ts_val = 'flight_2_hour_ts'
            elif pt_key.endswith('_4_hr'):
                ts_val = 'flight_4_hour_ts'
            elif pt_key.endswith('_2_day'):
                ts_val = 'flight_2_day_ts'
            elif pt_key.endswith('_4_day'):
                ts_val = 'flight_4_day_ts'
            elif pt_key.endswith('_day'):
                ts_val = 'flight_day_ts'
            elif pt_key.endswith('_wk'):
                ts_val = 'flight_wk_ts'
            elif pt_key.endswith('_hr'):
                ts_val = 'flight_ts'

            perspective_time = p + [ts_val]
            to_add_col = '_'.join([pt_key, 'delay_time'])
            to_merge_cols = p + [ts_val] + [to_add_col]
            to_merge_df = perspective_time_dfs[pt_key][to_merge_cols]
            merged_feature_df = merged_feature_df.merge(to_merge_df, how='left', left_on=perspective_time, right_on=perspective_time)

    # Delay only time
    perspective_time_dfs = delay_only_time_dfs
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key

            if pt_key.endswith('_2_hr'):
                ts_val = 'flight_2_hour_ts'
            elif pt_key.endswith('_4_hr'):
                ts_val = 'flight_4_hour_ts'
            elif pt_key.endswith('_2_day'):
                ts_val = 'flight_2_day_ts'
            elif pt_key.endswith('_4_day'):
                ts_val = 'flight_4_day_ts'
            elif pt_key.endswith('_day'):
                ts_val = 'flight_day_ts'
            elif pt_key.endswith('_wk'):
                ts_val = 'flight_wk_ts'
            elif pt_key.endswith('_hr'):
                ts_val = 'flight_ts'

            perspective_time = p + [ts_val]
            to_add_col = '_'.join([pt_key, 'delay_only_time'])
            to_merge_cols = p + [ts_val] + [to_add_col]
            to_merge_df = perspective_time_dfs[pt_key][to_merge_cols]
            merged_feature_df = merged_feature_df.merge(to_merge_df, how='left', left_on=perspective_time, right_on=perspective_time)

    # Delay count
    perspective_time_dfs = delay_count_dfs
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key

            if pt_key.endswith('_2_hr'):
                ts_val = 'flight_2_hour_ts'
            elif pt_key.endswith('_4_hr'):
                ts_val = 'flight_4_hour_ts'
            elif pt_key.endswith('_2_day'):
                ts_val = 'flight_2_day_ts'
            elif pt_key.endswith('_4_day'):
                ts_val = 'flight_4_day_ts'
            elif pt_key.endswith('_day'):
                ts_val = 'flight_day_ts'
            elif pt_key.endswith('_wk'):
                ts_val = 'flight_wk_ts'
            elif pt_key.endswith('_hr'):
                ts_val = 'flight_ts'

            perspective_time = p + [ts_val]
            to_add_col = '_'.join([pt_key, 'delay_count'])
            to_merge_cols = p + [ts_val] + [to_add_col]
            to_merge_df = perspective_time_dfs[pt_key][to_merge_cols]
            merged_feature_df = merged_feature_df.merge(to_merge_df, how='left', left_on=perspective_time, right_on=perspective_time)

    # Cancel count
    perspective_time_dfs = cancel_count_dfs
    for p_key in perspectives:
        for t_key in time_bins:
            p = perspectives[p_key]
            t = time_bins[t_key]

            pt_key = p_key + "_" + t_key

            if pt_key.endswith('_2_hr'):
                ts_val = 'flight_2_hour_ts'
            elif pt_key.endswith('_4_hr'):
                ts_val = 'flight_4_hour_ts'
            elif pt_key.endswith('_2_day'):
                ts_val = 'flight_2_day_ts'
            elif pt_key.endswith('_4_day'):
                ts_val = 'flight_4_day_ts'
            elif pt_key.endswith('_day'):
                ts_val = 'flight_day_ts'
            elif pt_key.endswith('_wk'):
                ts_val = 'flight_wk_ts'
            elif pt_key.endswith('_hr'):
                ts_val = 'flight_ts'

            perspective_time = p + [ts_val]
            to_add_col = '_'.join([pt_key, 'cancel_count'])
            to_merge_cols = p + [ts_val] + [to_add_col]
            to_merge_df = perspective_time_dfs[pt_key][to_merge_cols]
            merged_feature_df = merged_feature_df.merge(to_merge_df, how='left', left_on=perspective_time, right_on=perspective_time)

    # For delay/cancel count, fill NaN values with 0 (i.e. No delay/cancel flights)
    count_cols = [col_str for col_str in list(merged_feature_df.columns) if col_str.endswith('_count')]

    merged_feature_df[count_cols] = merged_feature_df[count_cols].fillna(0)

    merged_feature_df.to_csv(output_path)

    return
