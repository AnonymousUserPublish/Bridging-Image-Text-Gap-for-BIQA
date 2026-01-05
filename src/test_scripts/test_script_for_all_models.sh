#!/bin/bash

# ---- Wait 10 hours before starting ----


echo "Starting sequential execution now..."

# ---- List of scripts ----
# scripts=("./v13 ./test_all.sh" "./v12 ./test_all.sh" "./v13_1 ./test_all.sh" "./v13_2 ./test_all.sh" "./v14_1 ./test_all.sh" "./v14_2 ./test_all.sh")
scripts=("./v14_2 ./test_all.sh")

# ---- Run each script sequentially ----
for s in "${scripts[@]}"; do
    folder=$(echo $s | awk '{print $1}')
    script=$(echo $s | awk '{print $2}')

    echo "---------------------------------"
    echo "Running $script in $folder ..."
    
    # cd into the folder
    cd "$folder"

    # Run the script (must be executable)
    bash "$script"

    echo "$script finished."

    cd ..

done

echo "All scripts completed."
