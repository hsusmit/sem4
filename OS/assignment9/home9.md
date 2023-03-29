## OS Home Assignment 9

---

_1. Write a shell script that takes a list of names and displays all information in the password file, where login names are the members of the list._

```bash
#!/bin/bash

if [ $# -eq 0 ]
then
    echo "Please provide a list of login names as arguments."
    exit 1
fi

while [ $# -gt 0 ]
do
    login_name=$1
    shift
    echo "Information for login name $login_name:"
    grep -w "$login_name" /etc/passwd
    echo
done
```

---

_2. Write a shell script that periodically count the number of users logged into the system. Send the number of minutes at which to check as parameter._

```bash
#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Please provide a single argument indicating the number of minutes between checks."
    exit 1
fi

interval=$1
while true
do
    num_users=$(who | wc -l)
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Number of users logged in: $num_users"
    sleep ${interval}m
done
```

---

_3. Write a shell script to count the number of words of different length present in a given text._

```bash
#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Please provide a single argument indicating the file containing the text."
    exit 1
fi
file=$1
declare -A counts
while read -ra words; do
  for word in "${words[@]}"; do
    len=${#word}
    ((counts[len]++))
  done
done < "$file"
echo "Word Length    Count"
echo "-----------   -----"
for len in "${!counts[@]}"; do
  printf "%-12s %s\n" "$len" "${counts[$len]}"
done
```

---

_4. Write a shell script to count the frequency of different words used in a file. A shell script receives even number of filenames. Suppose four filenames are supplied then the first file should get copied into the second file, the third file should get copied into the fourth file, and so on. If odd number of filenames is supplied then no copying should take place and an error message should be displayed._

```bash
#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Please provide two arguments: the input file and the output file."
    exit 1
fi
input_file=$1
output_file=$2
if [ $(( $# % 2 )) -ne 0 ]
then
    echo "Error: odd number of filenames supplied."
    exit 1
fi
for ((i=1; i<=$#; i+=2))
do
    cp "${!i}" "${!((i+1))}"
done
declare -A word_counts
while read -r line
do
    for word in $line
    do
        ((word_counts[$word]++))
    done
done < "$input_file"
for word in "${!word_counts[@]}"
do
    printf "%s: %d\n" "$word" "${word_counts[$word]}" >> "$output_file"
done
```
