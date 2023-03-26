## OS Home Assignment 8

---

_1. Write a shell script to check if a given file (filename supplied as command line argument) is a regular file or not and find the total number of words, characters and lines in it._

```bash
#!/bin/bash

file=$1

if [ -f "$file" ]; then
    echo "Regular file"
    wc "$file"
else
    echo "Not a regular file"
fi

```

---

_2. Write a shell script which reads a directory name and compares the current directory with it (which has more files and how many more files)._

```bash
#!/bin/bash

dir=$1

if [ -d "$dir" ]; then
    num_files_dir=$(ls "$dir" | wc -l)
    num_files_cur=$(ls | wc -l)

    if [ $num_files_dir -gt $num_files_cur ]; then
        echo "$dir has more files than the current directory by $(expr $num_files_dir - $num_files_cur) files."
    elif [ $num_files_dir -lt $num_files_cur ]; then
        echo "The current directory has more files than $dir by $(expr $num_files_cur - $num_files_dir) files."
    else
        echo "Both directories have the same number of files."
    fi
else
    echo "Error: directory not found"
fi

```

---

_3. Write a shell script to check whether the given file is a blank file or not. If not found blank then display the contents of the file._

```bash
#!/bin/bash

file=$1

if [ -f "$file" ]; then
    if [ -s "$file" ]; then
        echo "Contents of $file:"
        cat "$file"
    else
        echo "$file is a blank file."
    fi
else
    echo "File not found."
fi

```

---

_4. Write a shell script to concatenate two files and count the number of characters, number of words and number of lines in the resultant file._

```bash
#!/bin/bash

file1=$1
file2=$2
outfile="output.txt"

cat "$file1" "$file2" > "$outfile"
echo "Number of characters: $(wc -m "$outfile" | cut -d ' ' -f 1)"
echo "Number of words: $(wc -w "$outfile" | cut -d ' ' -f 1)"
echo "Number of lines: $(wc -l "$outfile" | cut -d ' ' -f 1)"

```

---

_5. Write a shell script that accepts two directory names, say mca1 and mca2 as arguments and deletes those files in mca2 which have identical named files in mca1._

```bash
#!/bin/bash

dir1=$1
dir2=$2

if [ -d "$dir1" ] && [ -d "$dir2" ]; then
    for file in "$dir2"/*
    do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ -f "$dir1/$filename" ]; then
                rm "$file"
                echo "$file deleted."
            fi
        fi
    done
else
    echo "Error: directory not found"
fi

```
