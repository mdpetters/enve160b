+++
author = "Markus Petters"
maxtoclevel = 2
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = ["", ""]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = false
website_title = "ENVE-160B"
website_descr = "Course Website"
website_url   = "https://mdpetters.github.io/enve160b/"
+++

\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\concept}[1]{@@concept @@title **✎ Laboratory Report**@@ @@content #1 @@ @@}
\newcommand{\outline}[1]{@@outline @@title **✎ Lecture Outline**@@ @@content #1 @@ @@}
\newcommand{\note}[1]{@@note @@title **✎ Note**@@ @@content #1 @@ @@}
\newcommand{\learning}[1]{@@learning @@title **⏻ Learning Objectives**@@ @@content #1 @@ @@}
\newcommand{\outcomes}[1]{@@learning @@title **⏻ Learning Outcomes**@@ @@content #1 @@ @@}
\newcommand{\caution}[1]{@@warning @@title **⚠ Misconduct**@@ @@content #1 @@ @@}
\newcommand{\exercise}[1]{@@exercise @@title **⌨ Assignment**@@ @@content #1 @@ @@}

@def prepath = "enve160b"
