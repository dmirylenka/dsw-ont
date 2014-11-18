# Copyright 2014 University of Trento, Italy.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


################################################################################
## This is a file for writing one-off scripts, commenting them and messing up.
## It is intended to be run from the command line like this:
##
## >> python3.4 -m dswont.experiment.py
##
################################################################################

from dswont import category_explorer

category_explorer.run_selection_procedure(500)
