import streamlit as st
import pandas as pd
from datetime import timedelta
from datetime import datetime, timedelta


def function_1(df):
    # Get today's date for reference
    today = datetime.today().date()

    # Define the peak and off-peak times
    peak_start = datetime.combine(today, datetime.strptime("08:00", '%H:%M').time())
    peak_end = datetime.combine(today, datetime.strptime("10:00", '%H:%M').time())
    off_peak_start = datetime.combine(today, datetime.strptime("10:00", '%H:%M').time())
    off_peak_end = datetime.combine(today, datetime.strptime("18:00", '%H:%M').time())

    # Convert 'Start Time' and 'End Time' to datetime objects without the date component
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M').dt.time
    df['End Time'] = pd.to_datetime(df['End Time'], format='%H:%M').dt.time

    # Function to adjust time slots based on peak hours
    def adjust_schedule(row):
        start_time = row['Start Time']
        start_datetime = datetime.combine(today, start_time)
        
        # Adjust start times to fit a 10-minute frequency during peak hours
        if peak_start.time() <= start_time <= peak_end.time():
            minutes_since_peak_start = (start_datetime - peak_start).seconds // 60
            adjusted_start = peak_start + timedelta(minutes=(minutes_since_peak_start // 10) * 10)
        else:
            adjusted_start = start_datetime
        
        # Keep the same duration for the adjusted schedule
        end_datetime = datetime.combine(today, row['End Time'])
        duration = end_datetime - start_datetime
        adjusted_end = adjusted_start + duration
        
        # Return the new times
        return adjusted_start.strftime('%H:%M:%S'), adjusted_end.strftime('%H:%M:%S')

    # Apply the scheduling adjustments to 'Adjusted Start Time' and 'Adjusted End Time'
    df[['Adjusted Start Time', 'Adjusted End Time']] = df.apply(adjust_schedule, axis=1, result_type='expand')

    # Define in-city and out-city areas
    in_city_areas = [
        'Connaught Place', 'Delhi University', 'Kashmiri Gate', 'Rajiv Chowk',
        'Karol Bagh', 'Vikas Marg', 'Mayur Vihar', 'Hauz Khas', 'Saket',
        'Lajpat Nagar', 'Mandi House', 'India Gate', 'Paharganj',
        'New Delhi Railway Station', 'Jangpura', 'Chandni Chowk', 'Old Delhi Railway Station'
    ]

    out_city_areas = ['Dwarka Sector 21', 'Akshardham', 'Rohini Sector 15']

    # Function to reallocate buses from out-city to in-city areas during peak hours
    def reallocate_buses(row):
        start_time = datetime.combine(today, row['Start Time'])
        if row['Journey Type'] == 'Forward' and row['Source'] in out_city_areas:
            if peak_start.time() <= row['Start Time'] <= peak_end.time():
                old_source = row['Source']
                new_source = in_city_areas[row.name % len(in_city_areas)]
                new_destination = in_city_areas[(row.name + 1) % len(in_city_areas)]
                
                # Update the DataFrame
                df.at[row.name, 'Source'] = new_source
                df.at[row.name, 'Destination'] = new_destination
                
        return row

    # Apply the reallocation
    df = df.apply(reallocate_buses, axis=1)

    # Return the updated DataFrame
    return df

def function_2(df1,df2):
    short_routes = ['85', '73', '73LnkSTL', '71', 'RL77B', 'RL79', '33C', 'GL91', 'RL75']


    SHIFT_DURATION = timedelta(hours=8)  # Maximum 8-hour shift
    MORNING_TRIPS = 2
    EVENING_TRIPS = 2
    TRIP_DURATION = timedelta(minutes=80)  # Duration for each trip (forward or return)
    GAP_BETWEEN_TRIPS = timedelta(minutes=60)  # 1-hour gap between shifts

    # Define bus type time intervals
    BUS_TYPE_INTERVALS = {'AC': 0, 'Non-AC': 10, 'Electric': 15}  # Time gaps between bus types in minutes

    def adjust_shift_times(start_time, duration_minutes, gap_minutes=0):
        shift_start = pd.to_datetime(start_time) + timedelta(minutes=gap_minutes)
        shift_end = shift_start + timedelta(minutes=duration_minutes)
        return shift_start, shift_end

    def calculate_shift_times(start_time, bus_type, forward_duration=80, return_duration=80):
        forward_start, forward_end = adjust_shift_times(start_time, forward_duration, BUS_TYPE_INTERVALS[bus_type])
        return_start, return_end = adjust_shift_times(forward_end, return_duration, gap_minutes=10)
        return forward_start, forward_end, return_start, return_end

    def is_within_shift_limit(start_time, end_time):
        shift_start = pd.to_datetime(start_time)
        shift_end = pd.to_datetime(end_time)
        return shift_end - shift_start <= SHIFT_DURATION
    # Function to assign routes to employees and create round-trip schedules
    def assign_route_to_employee(busDF, employeeDF):
        assignments = []
        assigned_employees = set()  # Track assigned employees

        # For each bus, assign an employee
        for bus_index, bus in busDF.iterrows():
            route = bus['Route No.']
            start_time = bus['Start Time']

            # Filter short routes
            if route not in short_routes:
                continue

            for bus_type in ['AC', 'Non-AC', 'Electric']:  # Iterate through bus types
                # Find suitable crews
                available_crews = employeeDF[(employeeDF['Route Familiarity'] == route) & 
                                            (~employeeDF['EmpID'].isin(assigned_employees))]

                if not available_crews.empty:
                    crew = available_crews.iloc[0]  # Assign first available crew

                    # Calculate the shift times for the round trip (forward and return)
                    forward_start, forward_end, return_start, return_end = calculate_shift_times(start_time, bus_type)
                    
                    # Check if shifts are within the 8-hour limit
                    if is_within_shift_limit(forward_start, return_end):
                        # Forward trip assignment
                        assignments.append({
                            'EmpID': crew['EmpID'],
                            'Name': crew['Name'],
                            'Route No.': bus['Route No.'],
                            'Bus Type': bus_type,
                            'Source': bus['Source'],
                            'Destination': bus['Destination'],
                            'Trip Type': 'Forward',
                            'Shift Start': forward_start.strftime("%H:%M:%S"),
                            'Shift End': forward_end.strftime("%H:%M:%S")
                        })
                        
                        # Return trip assignment
                        assignments.append({
                            'EmpID': crew['EmpID'],
                            'Name': crew['Name'],
                            'Route No.': bus['Route No.'],
                            'Bus Type': bus_type,
                            'Source': bus['Destination'],
                            'Destination': bus['Source'],
                            'Trip Type': 'Return',
                            'Shift Start': return_start.strftime("%H:%M:%S"),
                            'Shift End': return_end.strftime("%H:%M:%S")
                        })

                        assigned_employees.add(crew['EmpID'])  # Mark employee as assigned for the day
                    
        return pd.DataFrame(assignments)

    def create_additional_trips(assignments):
        extended_assignments = []
        crew_trips = pd.DataFrame(assignments)
        
        for emp_id in crew_trips['EmpID'].unique():
            emp_trips = crew_trips[crew_trips['EmpID'] == emp_id]
            
            morning_start = pd.to_datetime('04:30:00')
            evening_start = pd.to_datetime('14:00:00')
            
            # Add additional trips if needed
            if len(emp_trips) < 4:
                remaining_trips = 4 - len(emp_trips)
                
                # Check the latest shift end
                latest_shift_end = pd.to_datetime(emp_trips['Shift End'].max())
                
                # Add morning trips if needed
                if remaining_trips > 0:
                    current_time = morning_start if latest_shift_end < morning_start else latest_shift_end
                    while remaining_trips > 0 and current_time.time() < pd.to_datetime('12:00:00').time():
                        forward_start, forward_end, return_start, return_end = calculate_shift_times(current_time, emp_trips['Bus Type'].iloc[0])
                        if is_within_shift_limit(forward_start, return_end):
                            extended_assignments.append({
                                'EmpID': emp_id,
                                'Name': emp_trips['Name'].iloc[0],
                                'Route No.': emp_trips['Route No.'].iloc[0],
                                'Bus Type': emp_trips['Bus Type'].iloc[0],
                                'Source': emp_trips['Source'].iloc[0],
                                'Destination': emp_trips['Destination'].iloc[0],
                                'Trip Type': 'Forward',
                                'Shift Start': forward_start.strftime("%H:%M:%S"),
                                'Shift End': forward_end.strftime("%H:%M:%S")
                            })
                            extended_assignments.append({
                                'EmpID': emp_id,
                                'Name': emp_trips['Name'].iloc[0],
                                'Route No.': emp_trips['Route No.'].iloc[0],
                                'Bus Type': emp_trips['Bus Type'].iloc[0],
                                'Source': emp_trips['Destination'].iloc[0],
                                'Destination': emp_trips['Source'].iloc[0],
                                'Trip Type': 'Return',
                                'Shift Start': return_start.strftime("%H:%M:%S"),
                                'Shift End': return_end.strftime("%H:%M:%S")
                            })
                            remaining_trips -= 1
                            current_time = return_end + GAP_BETWEEN_TRIPS
                        else:
                            break
                
                # Add evening trips if needed
                if remaining_trips > 0:
                    current_time = evening_start if latest_shift_end < evening_start else latest_shift_end
                    while remaining_trips > 0 and current_time.time() < pd.to_datetime('22:00:00').time():
                        forward_start, forward_end, return_start, return_end = calculate_shift_times(current_time, emp_trips['Bus Type'].iloc[0])
                        if is_within_shift_limit(forward_start, return_end):
                            extended_assignments.append({
                                'EmpID': emp_id,
                                'Name': emp_trips['Name'].iloc[0],
                                'Route No.': emp_trips['Route No.'].iloc[0],
                                'Bus Type': emp_trips['Bus Type'].iloc[0],
                                'Source': emp_trips['Source'].iloc[0],
                                'Destination': emp_trips['Destination'].iloc[0],
                                'Trip Type': 'Forward',
                                'Shift Start': forward_start.strftime("%H:%M:%S"),
                                'Shift End': forward_end.strftime("%H:%M:%S")
                            })
                            extended_assignments.append({
                                'EmpID': emp_id,
                                'Name': emp_trips['Name'].iloc[0],
                                'Route No.': emp_trips['Route No.'].iloc[0],
                                'Bus Type': emp_trips['Bus Type'].iloc[0],
                                'Source': emp_trips['Destination'].iloc[0],
                                'Destination': emp_trips['Source'].iloc[0],
                                'Trip Type': 'Return',
                                'Shift Start': return_start.strftime("%H:%M:%S"),
                                'Shift End': return_end.strftime("%H:%M:%S")
                            })
                            remaining_trips -= 1
                            current_time = return_end + GAP_BETWEEN_TRIPS
                        else:
                            break
        
        return extended_assignments

    # Example Usage
    busDF = df1
    crewDF = df2

    # Filter crews familiar with short routes
    reserved_short_crew = crewDF[crewDF['Route Familiarity'].isin(short_routes)]

    # Filter buses on short routes
    short_route_buses = busDF[busDF['Route No.'].isin(short_routes)]

    # Assign crews to short routes
    assignments_df = assign_route_to_employee(short_route_buses, reserved_short_crew)
    additional_trips = create_additional_trips(assignments_df)
    final_assignments = pd.concat([pd.DataFrame(assignments_df), pd.DataFrame(additional_trips)])
    return final_assignments


def function_3(df1,df2):
    bus_types = ['AC', 'Non-AC', 'Electric']
    long_routes = ['85 Cluster', '73 Cluster', '33A', '33LSTL', 'GL32']


    def calculate_shift_times(start_time, bus_type):
        forward_duration = pd.Timedelta(minutes=80)
        return_duration = pd.Timedelta(minutes=80)
        forward_start = pd.to_datetime(start_time)
        forward_end = forward_start + forward_duration
        return_start = forward_end + pd.Timedelta(minutes=10)  # 10-minute break
        return_end = return_start + return_duration
        return forward_start, forward_end, return_start, return_end

    def is_within_shift_limit(forward_start, return_end):
        total_shift_duration = return_end - forward_start
        max_shift_duration = pd.Timedelta(hours=8)
        return total_shift_duration <= max_shift_duration

    def assign_routes_and_schedule(busDF, crewDF):
        assignments = []
        assigned_employees = set()

        if busDF.empty or crewDF.empty:
            raise ValueError("Bus DataFrame or Employee DataFrame is empty.")
        
        reverse_route_map = {
            '85': '85',
            '85 Cluster': '85 Cluster',
            '73': '73',
            '73 Cluster': '73 Cluster',
            '73LnkSTL': '73LnkSTL',
            '71': '71',
            '72': '72',
            'RL77B': 'RL77B',
            'RL79': 'RL79',
            '33A': '33A',
            '33C': '33C',
            '33LSTL': '33LSTL',
            'GL32': 'GL32',
            'GL91': 'GL91',
            'RL75': 'RL75'
        }
        
        for bus_type in bus_types:
            for bus_index, bus in busDF.iterrows():
                route = bus['Route No.']
                start_time = bus['Start Time']
                bus_source = bus['Source']
                bus_destination = bus['Destination']
                shift_handover_point = bus['Handover Point']
                
                if route not in long_routes:
                    continue
                
                available_crews = crewDF[~crewDF['EmpID'].isin(assigned_employees)]
                if len(available_crews) < 2:
                    continue
                
                X = available_crews.iloc[0]
                Y = available_crews.iloc[1]
                
                forward_start, forward_end, return_start, return_end = calculate_shift_times(start_time, bus_type)
                
                if not is_within_shift_limit(forward_start, return_end):
                    continue

                # Assign Forward Trip
                assignments.append({
                    'EmpID': X['EmpID'],
                    'Name': X['Name'],
                    'Route No.': route,
                    'Bus Type': bus_type,
                    'Source': bus_source,
                    'Destination': bus_destination,
                    'Trip Type': 'Initial Forward',
                    'Shift Start': forward_start.strftime("%H:%M:%S"),
                    'Shift End': forward_end.strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })
                
                assignments.append({
                    'EmpID': Y['EmpID'],
                    'Name': Y['Name'],
                    'Route No.': reverse_route_map.get(route, route),
                    'Bus Type': bus_type,
                    'Source': bus_destination,
                    'Destination': bus_source,
                    'Trip Type': 'Initial Forward',
                    'Shift Start': forward_start.strftime("%H:%M:%S"),
                    'Shift End': forward_end.strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })

                # Handover
                assignments.append({
                    'EmpID': X['EmpID'],
                    'Name': X['Name'],
                    'Route No.': 'Handover',
                    'Bus Type': None,
                    'Source': None,
                    'Destination': None,
                    'Trip Type': 'Handover',
                    'Shift Start': forward_end.strftime("%H:%M:%S"),
                    'Shift End': (forward_end + pd.Timedelta(minutes=10)).strftime("%H:%M:%S"),
                    'Handover To': Y['Name'],
                    'Handover Point': shift_handover_point
                })
                
                assignments.append({
                    'EmpID': Y['EmpID'],
                    'Name': Y['Name'],
                    'Route No.': 'Handover',
                    'Bus Type': None,
                    'Source': None,
                    'Destination': None,
                    'Trip Type': 'Handover',
                    'Shift Start': forward_end.strftime("%H:%M:%S"),
                    'Shift End': (forward_end + pd.Timedelta(minutes=10)).strftime("%H:%M:%S"),
                    'Handover To': X['Name'],
                    'Handover Point': shift_handover_point
                })

                # Return Trip
                assignments.append({
                    'EmpID': Y['EmpID'],
                    'Name': Y['Name'],
                    'Route No.': route,
                    'Bus Type': bus_type,
                    'Source': bus_destination,
                    'Destination': bus_source,
                    'Trip Type': 'Return Forward',
                    'Shift Start': return_start.strftime("%H:%M:%S"),
                    'Shift End': return_end.strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })
                
                assignments.append({
                    'EmpID': X['EmpID'],
                    'Name': X['Name'],
                    'Route No.': reverse_route_map.get(route, route),
                    'Bus Type': bus_type,
                    'Source': bus_source,
                    'Destination': bus_destination,
                    'Trip Type': 'Return Forward',
                    'Shift Start': return_start.strftime("%H:%M:%S"),
                    'Shift End': return_end.strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })

                # Final Trip
                assignments.append({
                    'EmpID': X['EmpID'],
                    'Name': X['Name'],
                    'Route No.': route,
                    'Bus Type': bus_type,
                    'Source': bus_source,
                    'Destination': bus_destination,
                    'Trip Type': 'Final Forward',
                    'Shift Start': return_end.strftime("%H:%M:%S"),
                    'Shift End': (return_end + pd.Timedelta(minutes=80)).strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })
                
                assignments.append({
                    'EmpID': Y['EmpID'],
                    'Name': Y['Name'],
                    'Route No.': reverse_route_map.get(route, route),
                    'Bus Type': bus_type,
                    'Source': bus_destination,
                    'Destination': bus_source,
                    'Trip Type': 'Final Forward',
                    'Shift Start': return_end.strftime("%H:%M:%S"),
                    'Shift End': (return_end + pd.Timedelta(minutes=80)).strftime("%H:%M:%S"),
                    'Handover To': None,
                    'Handover Point': shift_handover_point
                })

                assigned_employees.update([X['EmpID'], Y['EmpID']])
            
        schedule_df = pd.DataFrame(assignments)
        return schedule_df



    # Example Usage
    busDF = df1
    crewDF = df2

    # Filter crews familiar with LONG routes
    reserved_long_crew = crewDF[crewDF['Route Familiarity'].isin(long_routes)]

    # Filter buses on LONG routes
    long_route_buses = busDF[busDF['Route No.'].isin(long_routes)]

    schedule_df = assign_routes_and_schedule(long_route_buses, reserved_long_crew)
    return schedule_df

# Mapping function names to actual functions
function_map = {
    "BUS SCHEDULING": (function_1, 1),
    "LINKED DUTY SCHEDULING": (function_2, 2),
    "UNLINKED DUTY SCHEDULING": (function_3, 2)  # Both require two files
}

# Streamlit app
st.set_page_config(page_title="CSV Processor App", page_icon="📊", layout="wide")  # Set page configuration

st.title("📊 CSV Processor App")

st.sidebar.header("Upload Your File(s) and Select Function")  # Sidebar header

# Function selection in sidebar
selected_function_name, num_files_required = st.sidebar.selectbox(
    "Select the function to apply",
    [(k, v[1]) for k, v in function_map.items()],
    format_func=lambda x: x[0]
)

uploaded_files = []

# Handling multiple file uploads
for i in range(num_files_required):
    file = st.sidebar.file_uploader(f"Upload CSV file {i+1}", type="csv", key=f"file_uploader_{i+1}")
    uploaded_files.append(file)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Your Name**")  # Add your name or branding

if None not in uploaded_files:
    # Load all files into dataframes
    if num_files_required == 1:
        dfs = [pd.read_csv(uploaded_files[0])]
    else:
        dfs = [pd.read_csv(uploaded_files[0]), pd.read_csv(uploaded_files[1])]

    # Get the selected function
    selected_function, num_args = function_map[selected_function_name]

    # Call the selected function with the loaded DataFrames
    if num_args == 1:
        result_df = selected_function(dfs[0])
    else:
        result_df = selected_function(dfs[0], dfs[1])

    # Display the result DataFrame
    st.subheader("Processed Data")
    st.dataframe(result_df)

    # Allow the user to download the result
    st.download_button(
        label="Download Processed Data as CSV",
        data=result_df.to_csv(index=False).encode('utf-8'),
        file_name="processed_data.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload all required files.")

