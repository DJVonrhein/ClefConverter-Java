# ClefConverter-Java
In progress Java version of the ClefConverter for my own experience

Step 1:
From project root directory, compile the main class:
```
mvn package
```

Step 2:
Run the jar that it created:
```
java -jar target/clef-converter-0.1.0.jar [path to sheet music image] bass
```
or
```
java -jar target/clef-converter-0.1.0.jar [path to sheet music image] treble
```

Step 3:
See the outputted sheet music in img/output.png
