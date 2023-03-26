## OS Lab Assignment 8

---

_1. Write a shell script, which reports names and sizes of all files in a directory (directory should be supplied as an argument to the shell script) whose size exceeds 100 bytes. The filenames should be printed in decreasing order of their sizes. The total number of such files should also be reported._

```bash
#!/bin/bash

dir=$1
count=0

if [ -d $dir ]; then
    find $dir -type f -size +100c -printf "%s %p\n" | sort -nr | while read -r size filename
    do
        echo "$filename: $size bytes"
        ((count++))
    done
    echo "Total number of files: $count"
else
    echo "Error: directory not found"
fi

```

---

_2. Write a shell script that shows the names of all the non-directory files in the current directory and calculates the sum of the size of them._

```bash
#!/bin/bash

sum=0
for file in *
do
    if [ -f "$file" ]; then
        echo "$file"
        size=$(stat -c %s "$file")
        ((sum+=size))
    fi
done
echo "Total size: $sum bytes"

```

---

_3. Write a shell script to list the name of files under the current directory that starts with a vowel._

```bash
#!/bin/bash

for file in [aeiou]* *[AEIOU]*
do
    if [ -f "$file" ]; then
        echo "$file"
    fi
done

```

---

_4. Write a shell script which receives two filenames as arguments and checks whether the two file's contents are same or not. If they are same then the second file should be deleted._

```bash
#!/bin/bash

if cmp -s "$1" "$2"; then
    rm "$2"
    echo "Files have same contents. Second file deleted."
else
    echo "Files have different contents."
fi

```
