#!/bin/sh
#  -*-Perl-*-  (for Emacs)    vim:set filetype=perl:  (for vim)
#======================================================================#
# Run the right perl version:
if [ -x /usr/local/bin/perl ]; then
    perl=/usr/local/bin/perl
elif [ -x /usr/bin/perl ]; then
	perl=/usr/bin/perl
else
    perl=`which perl | sed 's/.*aliased to *//'`
fi
exec $perl -x -S $0 "$@"     # -x: start from the following line
#======================================================================#
#! /Good_Path/perl -w
# line 17
#
# Name:   gkw_construct-parallel-output.pl
# Author: aps (A.P.Snodin@warwick.ac.uk)
# Date:   01-Dec-2008
# SVN: $Id: gkw_construct-parallel-output.pl 1031 2009-07-03 14:54:57Z $
# Description:
#
#  reconstruct the parallel.dat for spatially decomposed distribution

#----------------------------------------------------------------------------%

use strict;
use POSIX;

my %proc_map = ();
my ($proc_s,$ns,$ntot);
my @building_blocks=`ls par[0-9]*[0-9] | sort`; chomp(@building_blocks);
my $outputfile="parallel.dat";
my $nfiles;
my %handle = ();


foreach my $block (@building_blocks) {
    read_file($block);
}

while ( my ($key, $value) = each(%proc_map) ) {
    print "$key => $value\n";
}

$nfiles = keys( %proc_map ) ;
print "$nfiles\n";

open_files();
write_file();

#----------------------------------------------------------------------------%



#============================================================================%

sub write_file{
    my $line;
    open (OUT,">parallel.dat") or die ("Can not open $outputfile$!\n");

	foreach my $tot (1..$ntot) {
    
	    foreach my $file (1..$nfiles) {
		    my $j=$handle{$file};

			foreach my $i (1..$ns) {
			    seek($handle{$file},1,1) or die ("Problem in input file!\n");
		        $line = <$j>;
			    print OUT "$line";
            }

        }

    }
    close(OUT);
}

#============================================================================%

sub open_files{
    
    foreach my $i (1..$nfiles) {
        print "$proc_map{$i} \n";
	    open(my $fh,"<$proc_map{$i}") or die("error opening $proc_map{$i}\n");
        $handle{$i} = $fh;
		#print "$handle{$i}  : $fh \n";
	
		foreach my $var (my $a,my $b,my $c) {
		    seek($fh,1,1) or die ("Problem with input file\n");
	        $var=<$fh>;
        }

    }

}

#============================================================================%

sub read_file{
  
    my $filename = shift;
    print "Found $filename: ";
    open(FILE,"<$filename") or die("Could not open $filename\n");
    
	foreach my $var ($proc_s,$ns,$ntot) {
        seek(FILE, 1, 1) or die "Error in header of $filename$!\n";
        $var = <FILE>;
        $var =~ s/.*\=\W*//;
        chomp($var);
        print "$var, ";
    }

    close(FILE);
    $proc_map{$proc_s} = $filename;
    print "\n";

}

#============================================================================%

