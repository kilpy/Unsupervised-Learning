#!/usr/bin/perl -w
#use strict;
#use String::Approx 'amatch';
#use Text::LevenshteinXS qw(distance);
# use XML::Parser;
# use Term::ANSIColor qw(:constants);
# use IO::Socket::INET;
# use IO::Socket ":all";
# use Storable;
# use YAML::XS qw/LoadFile/;
# use List::MoreUtils qw/ uniq /;
# use Search::Elasticsearch;
# # use ElasticSearch;

my %HUSERGROUP=();
my %HGROUPUSER=();
my %HGREDGE=();
my %HUSEREDGES =();
my %HEDGE =();
my %SAMPLEDEDGE =();
my %SAMPLEDGROUP =();
my %SAMPLEDUSER= ();

$uf = "data/group-edges.csv";
open(FF,$uf) or die "cant open $uf\n";
while (<FF>){
	chomp;
	my @el = split (/,/);
	$HUSERGROUP{$el[0]}{$el[1]}=1;
	$HGROUPUSER{$el[1]}{$el[0]}=1;
	$HUSEREDGES{$el[0]}{"ingroup"}=0;
	$HUSEREDGES{$el[0]}{"outgroup"}=0;
	
} 
close(FF);
print STDERR "done reading file $uf\n\n";
$uf = "data/edges.csv";
open(FF,$uf) or die "cant open $uf\n";
while (<FF>){
	chomp;
	my @el = split (/,/);
	$u1= $el[0]; 	$u2= $el[1];
	$HEDGE{$u1}{$u2}=1;
	$HEDGE{$u2}{$u1}=1;
}
close(FF);
print STDERR "done reading file $uf\n\n";

######################################################################################

## sample groups, users
foreach $g (keys %HGROUPUSER){
	$nv  = keys %{$HGROUPUSER{$g}};
	if (rand()<0.2 && $nv>80) {
		$SAMPLEDGROUP{$g}=1;
		foreach $u (keys %{$HGROUPUSER{$g}}){
			$deg_u = keys %{$HEDGE{$u}};
			if (rand()<0.22 && $deg_u>20) { $SAMPLEDUSER{$u}=1;} else { $SAMPLEDUSER{$u}=0;} 
		}
	} 
	else {
		$SAMPLEDGROUP{$g}=0;
		foreach $u (keys %{$HGROUPUSER{$g}}) { $SAMPLEDUSER{$u}=0;}
	}
}


## sample edges
foreach $u1 (keys %HEDGE){
foreach $u2 (keys %{$HEDGE{$u1}}){
	if($u1>$u2) {next;}	
	@GR1= keys  %{$HUSERGROUP{$u1}}; @GR2= keys %{$HUSERGROUP{$u2}};
	foreach $g1 (@GR1){
		foreach $g2 (@GR2){
			$HGREDGE{$g1}{$g2} +=1; $HGREDGE{$g2}{$g1} +=1;
			if ($g1==$g2){
				$HUSEREDGES{$u1}{"ingroup"}+=1;
				$HUSEREDGES{$u2}{"ingroup"}+=1;
				if ($SAMPLEDUSER{$u1}==1  && $SAMPLEDUSER{$u2}==1  && rand()<=1.0) {$SAMPLEDEDGE{$u1}{$u2}=1;} 
			}
			else{
				$HUSEREDGES{$u1}{"outgroup"}+=1;
				$HUSEREDGES{$u2}{"outgroup"}+=1;	
				if ($SAMPLEDUSER{$u1}==1  && $SAMPLEDUSER{$u2}==1 && rand()<0.3) {$SAMPLEDEDGE{$u1}{$u2}=1;} 			
			}
		}
	}
	
	# linecount++;
	
}} 

######################################################################################
######################################################################################
######################################################################################

foreach $u (sort {$HUSEREDGES{$a}{"outgroup"}<=>$HUSEREDGES{$b}{"outgroup"}} keys %HUSEREDGES){
	print STDERR "user=$u  ingroup.edges=$HUSEREDGES{$u}{\"ingroup\"}  outgroup.edges=$HUSEREDGES{$u}{\"outgroup\"} \n";
	# getc();
}

# getc();

foreach $g (keys %HGREDGE){
	print STDERR "\nn\n\n group $g edge count to other groups:\n";
	foreach $h (sort {$HGREDGE{$g}{$b}<=>$HGREDGE{$g}{$a}} keys %{$HGREDGE{$g}}){
		print STDERR "  $h:$HGREDGE{$g}{$h}";
	}
	print STDERR "\n";
	# getc();
}

######################################################################################
$uf = ">data/edges_sampled.csv"; open(FF,$uf) or die "cant open $uf\n";
$ufm = ">data/edges_sampled_map.csv"; open(FFMAP,$ufm) or die "cant open $ufm\n";

my $countvetices = -1;
my %DICTMAP= ();
foreach $u (keys %SAMPLEDEDGE){
	foreach $v (keys %{$SAMPLEDEDGE{$u}}){
		if (!exists $DICTMAP{$u}) {$countvetices++; $DICTMAP{$u} =$countvetices;  }
		if (!exists $DICTMAP{$v}) {$countvetices++; $DICTMAP{$v} =$countvetices;  }
		print FF "$u,$v\n";
		print FFMAP "$DICTMAP{$u},$DICTMAP{$v}\n";
	}
}
close(FF);close(FFMAP);
