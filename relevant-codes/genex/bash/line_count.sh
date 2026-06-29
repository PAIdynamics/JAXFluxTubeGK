#!/bin/bash

CURRENT_PATH=$(pwd)
OUTPUT_DIR="./.line_count"

get_files() {
    # Get all files in the repo and exclude submodules
    # (the output has to be cropped afterwards)
    git ls-files -s | grep -v ^16 | cut -f2-
}

get_content() {
    # Returns the content of a file as a string, replaces line breaks
    # by markdown supported breaks
    cat -E $1 | sed -e "s/\$$/<br>/g"
}

get_total_lines() {
    # Returns the total number of lines that was written in a report file
    # (located on last line)
    grep -oP '\d+' $1 | tail -1
}

create_details() {
    # Create a collapsable details section in markdown
    CONTENT=$(get_content $1)
    echo "<details><summary>$1</summary><pre>$CONTENT</pre></details>"
}

mkdir -p $OUTPUT_DIR
get_files | xargs wc -l > "$OUTPUT_DIR/lines_total.txt"
get_files | grep '\.f90' | xargs wc -l > "$OUTPUT_DIR/lines_fortran.txt"
get_files | grep '\.pf'  | xargs wc -l > "$OUTPUT_DIR/lines_pfunit.txt"
get_files | grep '\.cxx\|\.hxx' | xargs wc -l > "$OUTPUT_DIR/lines_ccpp.txt"
get_files | grep '\.m$' | xargs wc -l > "$OUTPUT_DIR/lines_mathematica.txt"
get_files | grep '\.py' | xargs wc -l > "$OUTPUT_DIR/lines_python.txt"
get_files | grep -i 'cmake' | xargs wc -l > "$OUTPUT_DIR/lines_cmake.txt"
get_files | grep '\.txt' | grep -v 'CMakeLists' | grep -v 'params' \
          |& xargs wc -l > "$OUTPUT_DIR/lines_txtsources.txt"
cd $OUTPUT_DIR

LINES_TOTAL=$(get_total_lines lines_total.txt)
LINES_FORTRAN=$(get_total_lines lines_fortran.txt)
LINES_PFUNIT=$(get_total_lines lines_pfunit.txt)
LINES_CCPP=$(get_total_lines lines_ccpp.txt)
LINES_MATHEMATICA=$(get_total_lines lines_mathematica.txt)
LINES_PYTHON=$(get_total_lines lines_python.txt)
LINES_CMAKE=$(get_total_lines lines_cmake.txt)
LINES_TXTSOURCES=$(get_total_lines lines_txtsources.txt)
LINES_OTHER=$((LINES_TOTAL-LINES_FORTRAN-LINES_PFUNIT-LINES_CCPP\
               -LINES_MATHEMATICA-LINES_PYTHON-LINES_CMAKE-LINES_TXTSOURCES))

# Generate report
# NOTE: The empty lines are required for proper markdown display
FNAME="line_count_report.md"
echo "title: Line Count" > $FNAME
echo "" >> $FNAME
echo "# Line Count" >> $FNAME
echo "" >> $FNAME
echo "## Total number of lines in project: $LINES_TOTAL" >> $FNAME
echo "" >> $FNAME
echo "<br>" >> $FNAME
echo "## Breakdown:" >> $FNAME
echo "| Total       | $LINES_TOTAL   |" >> $FNAME
echo "|:------------|----------:|" >> $FNAME
echo "| Fortran     | $LINES_FORTRAN |" >> $FNAME
echo "| C/C++       | $LINES_CCPP |" >> $FNAME
echo "| Mathematica | $LINES_MATHEMATICA |" >> $FNAME
echo "| Python      | $LINES_PYTHON |" >> $FNAME
echo "| cmake       | $LINES_CMAKE |" >> $FNAME
echo "| pfUnit      | $LINES_PFUNIT |" >> $FNAME
echo "| txt sources | $LINES_TXTSOURCES |" >> $FNAME
echo "| other       | $LINES_OTHER |" >> $FNAME

echo "" >> $FNAME
echo "<br>" >> $FNAME
echo "## Details:" >> $FNAME
echo "" >> $FNAME
echo "$(create_details lines_total.txt)" >> $FNAME
echo "$(create_details lines_fortran.txt)" >> $FNAME
echo "$(create_details lines_pfunit.txt)" >> $FNAME
echo "$(create_details lines_ccpp.txt)" >> $FNAME
echo "$(create_details lines_mathematica.txt)" >> $FNAME
echo "$(create_details lines_python.txt)" >> $FNAME
echo "$(create_details lines_cmake.txt)" >> $FNAME
echo "$(create_details lines_txtsources.txt)" >> $FNAME
cd $CURRENT_PATH
