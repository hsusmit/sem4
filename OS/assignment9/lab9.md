## OS Lab Assignment 9

---

_1. A file called **list** consists of several words. Write a shell script which will receive a list of filenames, the first of which would be list. The shell script should report all occurrences of each word in **list** in the rest of the files supplied as arguments._

```bash
#!/bin/bash

if [ $# -eq 0]
then
	echo "please give  filenames"
	exit
fi
p= $1
shift
for var1 in $*
do
	for var2 in `cat $p`
		do
			echo "Occurence of $var2 in $var1 is : `grep -c $var2 $var1`"
		done
done
```

---

_2. Write a shell script which deletes all lines containing the word **UNIX** in the files supplied as arguments to this shell script._

```bash
#!/bin/bash

if[$# -eq 0]
then
	echo "please give  filenames"
	exit
fi
p="UNIX"
for file in $*
do
	sed "/$p/d" $file | tee tmp
	mv tmp $file
done
```

---

_3. Write a shell script which would receive a log name during execution, obtain information about it from password file and display this information on the screen in easily understandable format._

```bash
#!/bin/bash

if [ $# -ne 1 ]
then
    echo  "Enter your login name only"
    exit
fi
IFS=:
if [ $? -eq 0 ]
then
    set -- `grep -w "$1"  "/etc/passwd"
    echo "Login name:$1"
    echo "Password:$2"
    echo "UID:$3"
    echo "GUID:$4"
    echo "HOME:$5"
    echo "Login Shell:$7"
else
    echo "User Does not exist"
fi
```

---

_4. Write a shell script, which will receive either the filename or the filename with its full path during execution. This script should print information about the file as given by Is command and display it in an informative manner._

```bash
#!/bin/bash

if [ $# -ne 1 ]
then
    echo  "Enter a file name: "
    exit
fi
set -- `ls -1 $1`
echo "File permission: "$1
echo "Links: "$2
echo "Username: "$3
echo "Groupuser name: "$4
echo "Size: "$5
echo "Month: "$6
echo "Date: "$7
echo "Time: "$8
```
