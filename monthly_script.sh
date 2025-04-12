#!/bin/bash

# Define the expected number of days in each month (handles leap years)
days_in_month() {
    case $1 in
        01|03|05|07|08|10|12) echo 31 ;;
        04|06|09|11) echo 30 ;;
        02)
            # Check if leap year (divisible by 4 and not by 100 unless also by 400)
            if (( $2 % 4 == 0 && ( $2 % 100 != 0 || $2 % 400 == 0 ) )); then
                echo 29  # Leap year February
            else
                echo 28  # Non-leap year February
            fi
        ;;
    esac
}

# Directories
daily_dir="noaa_sst"
monthly_dir="monthly"

# Create the monthly output directory if it doesn't exist
mkdir -p "$monthly_dir"

# Loop through years (2000 to 2024) and months
for year in {2000..2024}; do
    for month in {01..12}; do
        output_file="${monthly_dir}/oisst_${year}${month}_monthly.nc"
        
        # Skip processing if the monthly file already exists
        if [ -f "$output_file" ]; then
            echo "Monthly file for $year-$month already exists in $monthly_dir. Skipping."
            continue
        fi

        # Determine expected number of days in the month
        expected_days=$(days_in_month $month $year)

        # Wait until all files for the month exist in the daily_dir
        while true; do
            file_count=$(ls ${daily_dir}/oisst-avhrr-v02r01.${year}${month}*.nc 2>/dev/null | wc -l)
            if [ "$file_count" -eq "$expected_days" ]; then
                break  # All files are available, proceed
            fi
            echo "Waiting for all files for $year-$month in $daily_dir... Found: $file_count / $expected_days"
            sleep 60  # Check again in 60 seconds
        done

        # Process the monthly mean using cdo and save the result in the monthly_dir
        cdo monmean -mergetime ${daily_dir}/oisst-avhrr-v02r01.${year}${month}*.nc "$output_file"
        echo "Processed monthly mean for $year-$month. Saved to $output_file."
    done
done

# Merge all monthly files into one final dataset using the --sortname option to fix warnings
final_file="oisst_monthly2.nc"
if [ ! -f "$final_file" ]; then
    cdo --sortname mergetime ${monthly_dir}/oisst_*_monthly.nc "$final_file"
    echo "Final merged file created: $final_file"
else
    echo "Final merged file already exists in $monthly_dir. Skipping merge."
fi
