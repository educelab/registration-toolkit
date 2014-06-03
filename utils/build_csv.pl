#!/usr/bin/perl
use warnings;
use strict;
use File::Find;

my $iterations = 200;
my $reg_suffix = "_Reg2010.tif";
my $tfm_suffix = ".tfm";
our @landmarks;
our %data;

find(\&wanted, '.');
sub wanted {
  if (/[.]ldm$/) {
    push @landmarks, $File::Find::name
  } elsif (/\d+-(.+)-(\d{4})[.]tif$/i) {
    $data{$1}->{$2} = $File::Find::name
  }
}

my %folder;
for (@landmarks) {
  /(.+)landmarks/;
  $folder{$1 . "transforms"}++;
  $folder{$1 . "registered"}++
}

for (keys %folder) {
  mkdir $_ if not defined -e $_
}


open(my $csv, ">", "registration_batch.csv");

for my $ldm (@landmarks) {
  $ldm =~ /\d+-(.+)-(\d{4})/;
  my ($page, $year) = ($1, $2);
  my $output_image = $data{$page}->{$year};
  $output_image =~ s/[.]tif$/$reg_suffix/i;
  $output_image =~ s#(\d{4})#$1/registered#;

  my $tfm_file = $ldm;
  $tfm_file =~ s/[.]ldm$/$tfm_suffix/i;
  $tfm_file =~ s/landmarks/transforms/;

  print $csv "$ldm, $data{$page}->{2010}, $data{$page}->{$year}, $output_image, $tfm_file, $iterations\n";
}
