The codebase uses ABAGAIL and BURLAP to perform MDP planning and learning.

Dataset :

Usage
------

Run from Ecplise IDE
	import the java project
	run SimpleGridExplorer as Java Application (e.g.)

Build Code
	include latest ABAGAIL.jar and burlap.jar in lib folder
	run Ant build to generate mdp_tests.jar

Run from CLI (mac / linux)
	cd mdp_tests
	1. java -cp mdp_tests.jar:lib/burlap.jar com.mdp.tests.GridExplorerValueIterator
	2. a) by default it uses testmaze_big.txt ;
	      there is another file testmaze_small.txt
	   b) java -cp mdp_tests.jar:lib/ABAGAIL.jar com.mdp.tests.MazeMDPTest
