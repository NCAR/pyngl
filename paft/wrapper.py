"""This script helps you wrap a Fortran subroutine so you
can call it from PyNGL."""

#
# This script asks you a bunch of questions about a Fortran routine
# that you want wrap and call from PyNGL. It then creates an interface
# C wrapper routine and a code snippet for registering the routine.
#
# There are some things that need to be added to this program:
# 
#  1. Recognition of work arrays
#  2. Ask if input arrays are not allowed to have missing values
#     (this is checked via "contains_missing").
#  3. Needs more testing with variables that don't have leftmost
#     dimensions.
#  4. More testing with scalars. I think there's something odd with these.
#  5. Better handling of array dimensions that are fixed. For example
#        for an array that is 2 x nx x ny, this program doesn't handle
#        the '2' part very well.
#

def write_fatal_block(wfile,msg,spaces="  "):
  wfile.write(spaces + '  printf("' + pyngl_name + ': fatal: ' + msg + '\\n");\n')
  wfile.write(spaces + '  Py_INCREF(Py_None);\n')
  wfile.write(spaces + '  return Py_None;\n')
  wfile.write(spaces + '}\n')

#
# Initialize some stuff.
#
args                = []
farg_names          = []
farg_cchar          = []
farg_types          = []
farg_ndims          = []
global_dsizes_names = []
input_var_names     = []
global_var_names    = []
global_var_types    = []
global_calling_char = []
global_dsizes_names_accum = []
various_var_names   = []
various_var_types   = []
work_array_names    = []
work_array_types    = []
index_names         = []
debug               = True
have_leftmost       = False

ntypes = ['numeric','double','float','integer','string','logical']
ctypes = ['double', 'double','float',    'int','string','logical']
reserved_names = ['i', 'ndims_leftmost', 'size_leftmost', 'arr']

#
# Set up class that will hold information on PyNGL input arguments.
#
class Argument:
  def __init__(self,arg_num,name,itype,sets_otype,ndims,dsizes=None,
               min_ndims=0,dsizes_names=None,dsizes_names_str="",
               has_missing=False):
    self.arg_num    = arg_num       # Number of argument, -1 indicates
                                    # return value.
    self.name       = name
    self.ntype      = ntypes[itype]
    self.ctype      = ctypes[itype]
    self.sets_otype = sets_otype
    self.ndims      = ndims
    self.min_ndims  = min_ndims
    if ndims == 1 and dsizes[0] == 1:
      self.is_scalar = True
    else:
      self.is_scalar = False
#
# These are variable names that we use in the C code to hold
# number of dimensions, dimension sizes, types, missing values, and
# temporary names for various input arguments. Note that not all of
# these names will necessarily be used in the code. They are
# conditional on several factors, like whether the type is numeric,
# and whether the array has leftmost dimensions.
#
    self.arr_name     = "arr_" + name
    self.tmp_name     = "tmp_" + name
    self.ndims_name   = "ndims_" + name
    self.type_name    = "type_" + name
    self.size_name    = "size_" + name
    self.dsizes_name  = "dsizes_" + name
    self.has_msg_name = "has_missing_" + name
    self.msg_name     = "fill_value_"     + name
    self.index_name   = "index_" + name
    if self.arg_num == -1:
      self.ret_name   = "ret_" + name

    if dsizes != None:
      self.dsizes      = dsizes
    if dsizes_names != None:
      self.dsizes_names     = dsizes_names
      self.dsizes_names_str = dsizes_names_str
    self.has_missing = has_missing
#
# Set up instructions on how to print an instance of this class.
#
  def __str__(self):
    if self.arg_num >= 0:
      str1 = "Name of argument " + str(self.arg_num) + " is '" +  \
              self.name + "'\n"
    else:
      str1 = "Name of return argument is '" + self.name + "'\n"
    str1 = str1 + "PyNGL type is " + self.ntype + "\n"
    str1 = str1 + "    C type is " + self.ctype + "\n"
    if self.has_missing:
      str1 = str1 + "    This variable can contain a missing value.\n"
    if self.ndims > 0:
      if self.is_scalar:
        str1 = str1 + "  This variable is a scalar\n"
      else:
        str1 = str1 + "  Number of dimensions is " + str(self.ndims) + "\n"
        str1 = str1 + "  Name of variable that holds # of dimemsions is " + \
               self.ndims_name + "\n"
        str1 = str1 + "  Name of variable that holds dimemsion sizes is " + \
               self.dsizes_name + "\n"
        for i in range(self.ndims):
          if self.dsizes[i] > 1:
            str1 = str1 + "  Dimension # " + str(i) + " is length " +\
                          str(self.dsizes[i]) + "\n"
          else:
            str1 = str1 + "  Dimension # " + str(i) + " is variable\n"
    else:
      str1 = str1 + "  Number of dimensions is variable\n"

    if self.min_ndims > 0:
      str1 = str1 + "  Number of minimum dimensions is " + \
                    str(self.min_ndims) + "\n"
      for j in range(self.min_ndims):
        str1 = str1 + "  Dimension " + self.ndims_name + "-" + \
                      str(int(self.min_ndims-j)) + " is " + \
                      self.dsizes_names[j] + "\n"
    return str1

#
# Import some stuff.
#
import sys
from string import *

#
# Begin asking for input on function/procedure.
#
print "\nIn order to wrap your Fortran subroutine, I need"
print "to ask you a few questions.\n"

#
# Is this a function or procedure?
#

forp = raw_input("Is this going to be a function (f) or procedure (p)? [f] ")
if (lower(forp) == "p"):
  isfunc       = False
  wrapper_type = "procedure"
else:
  isfunc       = True
  wrapper_type = "function"

#
# Get Fortran, PyNGL, and wrapper names of function or procedure.
#

valid = False
while not valid:
  pyngl_name = raw_input("\nEnter PyNGL name of your " + wrapper_type + ": ")
  if pyngl_name == "":
    print "Invalid name for PyNGL function or procedure, reenter."
  else:
    valid = True

pyngl_nameP = pyngl_name + "_P"     # Name of wrapper.

valid = False
while not valid:
  fortran_name = raw_input("\nEnter Fortran name of your " + wrapper_type + \
                           ": ")
  if fortran_name == "":
    print "Invalid name for Fortran subroutine, reenter."
  else:
    valid = True

valid = False
while not valid:
  wrapper_name = raw_input("\nEnter name of wrapper C file (without '.c') " +\
                           ": ")
  if wrapper_name == "":
    print "Invalid name for wrapper name, reenter."
  else:
    valid = True

#
# How many input arguments are there?
#

valid = False
while not valid:
  rinput = raw_input('\nHow many input arguments are there for ' + \
                     pyngl_name + '? ')
  try:
    num_args = int(rinput)
    if num_args < 0:
      print "Invalid number of arguments: ",num_args
      print "Reenter."
    elif num_args == 0:
      print "This script is for routines that contain arguments. Reenter."
    else:
      valid = True
  except:
      print "Must enter an integer > 0, reenter."

print "\nI need to ask you some questions about each input argument."

#
# Loop across the number of input arguments and get information
# about each one.
#

num_input_has_missing = 0       # Counter for how many input
                                # variables can have missing values.
for i in range(num_args):

#
# Get name of argument.
#

  valid = False
  while not valid:
    name = raw_input("\nWhat is the name of argument # " + str(i) + "? ")
    if name == "":
      print "Invalid name, reenter."
    elif name in reserved_names or name in global_var_names:
      print "Name already in use, reenter."
    else:
      valid = True

#
# Get type of argument.
#

  print "What type is '" + name + "'? [0]"
  for j in range(len(ntypes)):
    print "   ",j,":",ntypes[j]
   
  valid = False
  while not valid:
    rinput = raw_input()
    if rinput == "":
      itype = 0
      valid = True
    else:
      try:
        itype = int(rinput)
        if (itype < 0) or (itype >= len(ntypes)):
          print "Invalid type, reenter."
        else:
          valid = True
      except:
        print "Invalid type, reenter."
#
# Store this variable and its type in a list of variables we are
# keeping track of.
#
  input_var_names.append(name)
  global_var_names.append(name)
  global_var_types.append(itype)
  global_calling_char.append("")

#
# If this input argument is numeric, then ask if its type
# determines the output type of the return value.
#
  if isfunc and itype == 0:
    rinput = raw_input("Does the type of " + name + " determine the type of the output? (y/n) [n] ")
  if (lower(rinput) == "y"):
    sets_otype = True
  else:
    sets_otype = False

#
# Ask about a missing value.
#
  rinput = raw_input("Can " + name + " have a missing value? (y/n) [n] ")
  if (lower(rinput) == "y"):
    has_missing = True
    num_input_has_missing += 1

    global_var_names.append("fill_value_" + name)
    global_var_types.append(ctypes.index('double'))
    global_calling_char.append("&")
  else:
    has_missing = False

#
# Get dimension sizes.
#

  print "How many dimensions does '" + name + "' have?"

  valid = False
  while not valid:
    rinput = raw_input("Enter <return> for variable dimensions: [0] ")
    if rinput == "":
      ndims = 0
      valid = True
    else:
      try:
        ndims = int(rinput)
        if ndims >= 0:
          valid = True
        else:
          print "Invalid number of dimensions, reenter."
      except:
        print "Must enter an integer, reenter."

  dsizes       = None
  dsizes_names = None
  min_ndims    = 0
  dsizes_names_str = ""

  if ndims > 0:
#
# Our array is a fixed number of dimensions.
#
    print "Enter dimension size of each dimension"
    print "(Hit <return> for variable dimension size)"
    dsizes = []
    for j in range(ndims):
      valid = False
      while not valid:
        rinput = raw_input("Size for dimension # " + str(j) + \
                           " of argument '" + name + "': ")
        if rinput == "":
          valid = True
          dsizes.append(0)
        else:
          try:
            if int(rinput) >= 0:
              valid = True
              dsizes.append(int(rinput))
            else:
              print "Invalid size for dimension, reenter."
          except:
            print "Must enter an integer, reenter."
  else:
#
# Our array is a variable number of dimensions. That is, it can contain
# leftmost dimensions.
#
    have_leftmost = True
    index_names.append("index_" + name)
#
# Since this input variable can have variable leftmost dimensions, we
# will create a temporary variable to hold the pointer into the original
# array. This is not necessary, but just helpful.
#
    global_var_names.append("tmp_" + name)
    global_var_types.append(itype)
    global_calling_char.append("")

#
# Get the minimum dimension size required.
#

    valid = False
    while not valid:
      rinput = raw_input("How many dimensions does the Fortran routine expect for this variable? ")
      try:
        min_ndims = int(rinput)
        if min_ndims > 0:
          valid = True
        else:
          print "Invalid number of dimensions, reenter."
      except:
        print "Must enter an integer > 0."

#
# Get the names of each non-leftmost dimension size.
#
  if not (ndims == 1 and dsizes[0] == 1):
    dsizes_names     = []
    dsizes_names_str = ""
    print "What are the names of each of these dimensions?"      
    if global_dsizes_names != []:
      print "You can use these existing names if they apply: " + \
             str(global_dsizes_names)

    for j in range(max(ndims,min_ndims)):
      if ndims > 0:
        dsizes_names.append(raw_input("Name of dimension " + str(j) + " : "))
      else:
        dsizes_names.append(raw_input("Name of dimension ndims_" + name + \
                                    "-" + str(int(min_ndims-j)) + " : "))
      if not dsizes_names[j] in global_dsizes_names: 
        global_dsizes_names.append(dsizes_names[j])
      if not dsizes_names[j] in global_var_names: 
        global_var_names.append(dsizes_names[j])
        global_var_types.append(ctypes.index('int'))
        global_calling_char.append("&")
#
# Create string for variable that will hold the size of these dimensions
# (rightmost dimensions for variables that have leftmost dimensions).
#
# For example, if the dimensions are nx, ny, and nz, then
# a variable called "nxnynz" will be created that is equal to 
# nx * ny * nz. 
#
      dsizes_names_str = dsizes_names_str + dsizes_names[j]

    if not dsizes_names_str in global_dsizes_names: 
      global_dsizes_names.append(dsizes_names_str)
    if not dsizes_names_str in global_var_names: 
      global_var_names.append(dsizes_names_str)
      global_var_types.append(ctypes.index('int'))
      global_calling_char.append("&")
#
# With all this information, create an instance of the Argument class.
#

  args.append(Argument(i,name,itype,sets_otype,ndims,dsizes,min_ndims, \
                       dsizes_names,dsizes_names_str,has_missing))

#
# Get information on the return value, if a function.
#
if isfunc:
  valid = False
  while not valid:
    ret_name = raw_input("\nWhat is the name of the return value? ")
    if ret_name == "":
      print "Invalid name, reenter."
    elif ret_name in reserved_names or ret_name in global_var_names:
      print "Name already in use, reenter."
    else:
      valid = True

#
# Get type of argument.
#
  print "What type is '" + ret_name + "'? [0] "
  for j in range(len(ntypes)):
    print "   ",j,":",ntypes[j]
   
  valid = False
  while not valid:
    rinput = raw_input()
    if rinput == "":
      ret_itype = 0
      valid = True
    else:
      try:
        ret_itype = int(rinput)
        if (ret_itype < 0) or (ret_itype >= len(ntypes)):
          print "Invalid type, reenter."
        else:
          valid = True
      except:
        print "Must enter an integer, reenter."

  global_var_names.append(ret_name)
  global_var_types.append(ret_itype)
  global_calling_char.append("")

#
# Ask about a return missing value.
#
  rinput = raw_input('Can ' + ret_name + ' contain missing values? (y/n) [n] ')
  if (lower(rinput) == "y"):
    ret_has_missing = True
  else:
    ret_has_missing = False

#
# Find out which (if any) input variable the return value is dependent on.
# Only do this if there's at least one input value that can contain a
# missing value.
#
  if ret_has_missing and (num_input_has_missing > 0):
    rinput = raw_input("Is the return missing value based on any input missing values? (y/n) [y] ")
    if (lower(rinput) == "n"):
      ret_msg_depend_input = False
    else:
      ret_msg_depend_input = True

    if ret_msg_depend_input:  
      print "Select the input variable whose missing value determines"
      print "the missing value for the return variable:"
  
      for j in range(len(args)):
        if(args[j].has_missing):
          print "   ",j,":",args[j].name
   
      valid = False
      while not valid:
        rinput = raw_input()
        try:
          ret_msg_depend_index = int(rinput)
          if (ret_msg_depend_index < 0) or \
             (ret_msg_depend_index >= len(args)):
            print "Invalid entry, reenter."
          else:
            valid = True
        except:
          print "Must enter an integer, reenter."

#
# Find out if the size of the output is the exact same as any of the
# input.
#
  rinput = raw_input("Do the dimension sizes of the return value have to be the exact same as any of the input values? (y/n) [n] ")
  if (lower(rinput) == "y"):
    ret_size_depend_input = True
  else:
    ret_size_depend_input = False

  if ret_size_depend_input:  
    print "Select the input variable whose dimension sizes determine"
    print "the dimension sizes for the return variable:"
  
    for j in range(len(args)):
      print "   ",j,":",args[j].name
   
    valid = False
    while not valid:
      rinput = raw_input()
      try:
        ret_size_depend_index = int(rinput)
        if (ret_size_depend_index < 0) or \
           (ret_size_depend_index >= len(args)):
          print "Invalid entry, reenter."
        else:
          valid = True
      except:
        print "Must enter an integer, reenter."

#
# If the dimension sizes of the return variable are not dependent on
# an input variable, then query for them here.
#
  if ret_size_depend_input:  
#
# Assign dimension size information of input variable to the
# output variable.
#
    ret_ndims            = args[ret_size_depend_index].ndims
    ret_min_ndims        = args[ret_size_depend_index].min_ndims
    if args[ret_size_depend_index].ndims > 0:
      ret_dsizes           = args[ret_size_depend_index].dsizes
    else:
      ret_dsizes       = None
      index_names.append("index_" + ret_name)

    if not args[ret_size_depend_index].is_scalar:
      ret_dsizes_names     = args[ret_size_depend_index].dsizes_names
      ret_dsizes_names_str = args[ret_size_depend_index].dsizes_names_str
    else:
      ret_dsizes_names = None
      ret_dsizes_names_str = ""
  else:
#
# Get dimension sizes.
#
    print "How many dimensions does " + ret_name + " have?"

    valid = False
    while not valid:
      rinput = raw_input("Hit <return> for variable dimensions: ")
      if rinput == "":
        ret_ndims = 0
        valid = True
      else:
        try:
          ret_ndims = int(rinput)
          if ret_ndims < 0:
            print "Invalid number of dimensions, reenter."
          else:
            valid = True
        except:
          print "Must enter an integer >= 0, reenter."

    ret_dsizes       = None
    ret_dsizes_names = None
    ret_min_ndims    = 0
    ret_dsizes_names_str = ""

    if ret_ndims > 0:
      print "Enter dimension size of each dimension"
      print "(Hit <return> for variable dimension size)"
      ret_dsizes = []
      for j in range(ret_ndims):
        valid = False
        while not valid:
          rinput = raw_input("Size for dimension # " + str(j) + " of return value: ")
          if rinput == "":
            ret_dsizes.append(0)
            valid = True
          else:   
            try:      
              if int(rinput) < 0:
                print "Invalid size for dimension, reenter."
              else:
                ret_dsizes.append(int(rinput))
                valid = True          
            except:
              print "Must enter an integer, reenter."
    else:
      have_leftmost = True
      index_names.append("index_" + ret_name)

#
# Get the minimum dimension size required.
#
      valid = False
      while not valid:
        rinput = raw_input("How many dimensions does the Fortran routine expect for the return value? ")
        try:
          ret_min_ndims = int(rinput)
          if ret_min_ndims <= 0:
            print "Invalid number of dimensions, reenter."
          else:
            valid = True
        except:
          print "Must enter an integer > 0, reenter."

#
# Get the names of each non-leftmost dimension size.
#
    if not (ret_ndims == 1 and ret_dsizes[0] == 1):
      ret_dsizes_names     = []
      ret_dsizes_names_str = ""

      print "What are the names of each of these dimensions?"      
      if global_dsizes_names != []:
        print "You can use these existing names if they apply: " + \
              str(global_dsizes_names)
      for j in range(max(ret_ndims,ret_min_ndims)):
        if ret_ndims > 0:
          ret_dsizes_names.append(raw_input("Name of dimension " + str(j) + \
                                            " : "))
        else:
          ret_dsizes_names.append(raw_input("Name of dimension ndims_" + \
                     ret_name + "-" + str(int(ret_min_ndims-j)) + " : "))
          if not ret_dsizes_names[j] in global_dsizes_names: 
            global_dsizes_names.append(ret_dsizes_names[j])
          if not ret_dsizes_names[j] in global_var_names: 
            global_var_names.append(ret_dsizes_names[j])
            global_var_types.append(ctypes.index('int'))
            global_calling_char.append("&")
#
# Create string for variable that will hold the size of these 
# minimum dimensions.
#
          ret_dsizes_names_str = ret_dsizes_names_str + ret_dsizes_names[j]

        if not ret_dsizes_names_str in global_dsizes_names: 
          global_dsizes_names.append(ret_dsizes_names_str)
        if not ret_dsizes_names_str in global_var_names: 
          global_var_names.append(ret_dsizes_names_str)
          global_var_types.append(ctypes.index('int'))
          global_calling_char.append("&")
  
#
# With this information, create an instance of the Argument class for
# the return value.
#
  ret_arg = Argument(-1,ret_name,ret_itype,False,ret_ndims,ret_dsizes,
                     ret_min_ndims,ret_dsizes_names,ret_dsizes_names_str,
                     ret_has_missing)

#
# Get information about how Fortran function is to be called and what
# types its input variables are.
#


valid = False
while not valid:
  rinput = raw_input('\nHow many input arguments are there for the Fortran routine ' + fortran_name + '? ')
  try:
    fnum_args = int(rinput)
    if fnum_args <= 0:
      print "Invalid number of dimensions for Fortran routine."
      print "There should be at least one dimension, reenter."
    else:
      valid = True
  except:
    print "Must enter an integer > 0, reenter."

#
# Loop through the Fortran arguments and get the name and type.
#
for i in range(fnum_args):
  print "What is the name of argument # " + str(i) + " ?"
  print "You can select one of these existing names, or supply your own."
  print "It is highly recommended you select 'tmp_xxxx' rather than"
  print "'xxxx' if 'xxxx' contains variable leftmost dimensions."

  for j in range(len(global_var_names)):
    print "   ",j,":",global_var_names[j]

  rinput = raw_input()
  valid = False
  from_list = False
  while not valid:
    try:
      ii = int(rinput)
      if (ii >= 0) or (ii < len(global_var_names)):
        valid = True
        farg_names.append(global_var_names[ii])
        farg_cchar.append(global_calling_char[ii])
        from_list = True
      else:
        print "Invalid integer, reenter."
    except:
      farg_names.append(rinput)
      various_var_names.append(rinput)
      valid = True
      rinput = raw_input("Does an '&' need to be prepended when passing '" +\
                         rinput + "' to the Fortran routine? (y/n) [n] ")
      if (lower(rinput) == "y"):
        farg_cchar.append('&')
      else:
        farg_cchar.append('')

  valid = False
  while not valid:
    if from_list:
      farg_types.append(global_var_types[ii])
      valid = True
    else:
      print "What type is '" + farg_names[i] + "'? [1] "
      for j in range(len(ctypes)):
        print "   ",j,":",ctypes[j]
   
      rinput = raw_input()
      if rinput == "":
        farg_types.append(1)
        various_var_types.append(1)
        valid = True
      else:
        try:
          if (int(rinput) < 0) or (int(rinput) >= len(ctypes)):
            print "Invalid type, reenter."
          else:
            farg_types.append(int(rinput))
            various_var_types.append(int(rinput))
            valid = True
        except:
          print "Must enter an integer, reenter."

#
# Print information about each argument for debugging purposes.
#
if debug:
  for i in range(len(args)):
    print args[i]

  if isfunc:
    print ret_arg

#
# Open wrapper file and start writing to it, but first
# make sure this is acceptable.
#

print 'I will be creating the files ' + wrapper_name + 'P.c and ' + \
      'wrapper_' + pyngl_name + '.c.'
print 'The contents of wrapper_' + pyngl_name + '.c should be copied over to fplibmodule.c.'

okay = raw_input("Is this okay? (y/n) [y] ")
if (lower(okay) == "n"):
  print "Bye!"
  sys.exit()

#
# Open the files. The first one (w1file) is for actually wrapping the
# Fortran code, and the second one (w2file) is for registering the
# function or procedure. The second file is for temporary use only: you
# should copy its contents to "wrapper.c" where all the built-in 
# functions and procedures are registered.
#
w1file = open(wrapper_name+'P.c','w')
w2file = open('wrapper_' + wrapper_name+'.c','w')

#
# Start writing information to the main wrapper code.
#

#---------------------------------------------------------------------
#
# Write out the include files needed.  [None needed for Python wrappers?]
#
#---------------------------------------------------------------------
# w1file.write('#include <stdio.h>\n')
#

#---------------------------------------------------------------------
#
# Write out prototype for Fortran routine.
#
#
#  extern void NGCALLF(xxx,XXX)(double *, int*, etc *);
#
#---------------------------------------------------------------------
w1file.write("extern void NGCALLF("+ lower(fortran_name) + "," +
             upper(fortran_name) + ")(")
for i in range(len(farg_types)):
  if i == 0:
    w1file.write(ctypes[farg_types[i]] + " *")
  else:
    w1file.write(", " + ctypes[farg_types[i]] + " *")
w1file.write(");\n\n")

#---------------------------------------------------------------------
#
# Write out first line of PyNGL wrapper routine.
#
# PyObject *fplib_wrf_rh(PyObject *self, PyObject *args) {
#
#---------------------------------------------------------------------
w1file.write("PyObject *fplib_" + pyngl_name + "(PyObject *self, PyObject *args)\n{\n")

#---------------------------------------------------------------------
#
# Write out the declarations for the input argument variables.
#
#---------------------------------------------------------------------

#---------------------------------------------------------------------
#
# First write code to declare variable itself along with its type.
#
#---------------------------------------------------------------------
for i in range(len(args)):
  w1file.write("/*\n")
  w1file.write(" * Argument # " + str(i) + "\n")
  w1file.write(" */\n")

  w1file.write("  PyObject *" + args[i].arr_name + " = NULL;\n")
  if args[i].ndims == 0:
    w1file.write("  " + args[i].ctype + " *" + args[i].name + ", *" + \
                 args[i].tmp_name + ";\n")
  else:
    w1file.write("  " + args[i].ctype + " *" + args[i].name + ";\n")

#---------------------------------------------------------------------
#
# Write out dimension information.
#
# int ndims_x, dsizes_x[...];
#
#---------------------------------------------------------------------
  if args[i].ndims == 0:
    w1file.write("  int " + args[i].ndims_name + ", *" + \
                 args[i].dsizes_name + ";\n")
  else:
#
# We only need to include the dimension sizes if one of the sizes
# is unknown (represented by being set to '0').
#
    if 0 in args[i].dsizes:
      w1file.write("  int " + args[i].dsizes_name + "[" + str(args[i].ndims) + "];\n")

#---------------------------------------------------------------------
#
# Include missing value variables for each variable that can contain
# missing values.
#
# int has_missing_x;
# double fill_value_x;
#
#---------------------------------------------------------------------
  if args[i].has_missing:
    w1file.write("  int " + args[i].has_msg_name + ";\n")
    w1file.write("  " + args[i].ctype + " " + args[i].msg_name + ";\n")

if isfunc:
#---------------------------------------------------------------------
#
# Write out declarations for return variable information.
#
#---------------------------------------------------------------------
  w1file.write("/*\n")
  w1file.write(" * Return variable\n")
  w1file.write(" */\n")

  w1file.write("  PyArrayObject *ret_" + ret_arg.name + ";\n")
  w1file.write("  " + ret_arg.ctype + " *" + ret_arg.name + ";\n")

#---------------------------------------------------------------------
#
# Write out dimension information.
#
# If the dimension sizes are identical to one of the input variables'
# dimension sizes, then nothing is needed here.
#---------------------------------------------------------------------
  if not ret_size_depend_input:
    w1file.write("  int " + ret_arg.ndims_name + ", *" + \
                 ret_arg.dsizes_name + ";\n")

#---------------------------------------------------------------------
#
# Include missing value variables for each variable that can contain
# missing values.
#
# int has_missing_x;
# double fill_value_x;
#
#---------------------------------------------------------------------
  if ret_arg.has_missing:
    w1file.write("  int " + ret_arg.has_msg_name + ";\n")
    w1file.write("  " + ret_arg.ctype + " " + ret_arg.msg_name + ";\n")

#
# Define some PyArray objects for input and output.
#
w1file.write("""
/*
 * Define some PyArray objects for input and output.
 */
""")
w1file.write("  PyArrayObject *arr;\n")

#---------------------------------------------------------------------
#
# Write out code for other random variables, like dimension sizes.
#
#---------------------------------------------------------------------
w1file.write("""
/*
 * Various
 */
""")
if global_dsizes_names != []:
  w1file.write("  int ")
#
# Write out the various dimension size variables we've been collecting
# into the global_dsizes_names array.
#
  for i in range(len(global_dsizes_names)):
    if i == (len(global_dsizes_names)-1):
      w1file.write(global_dsizes_names[i] + ";\n")
    else:
      w1file.write(global_dsizes_names[i] + ", ")
#---------------------------------------------------------------------
#
# For any variable that has leftmost dimensions, we need a corresponding
# "index_xxx" variable.
#
#---------------------------------------------------------------------
  w1file.write("  int ")
  for i in range(len(index_names)):
    if i == (len(index_names)-1):
      w1file.write(index_names[i] + ";\n")
    else:
      w1file.write(index_names[i] + ", ")

#
# Write out the various work arrays and extra arguments we have to
# pass into Fortran array.
#
if various_var_names != []:
  for i in range(len(various_var_names)):
    w1file.write("  " + ctypes[various_var_types[i]] + " " + \
                 various_var_names[i] + ";\n")

if work_array_names != []:
  for i in range(len(work_array_names)):
    w1file.write("  " + ctypes[work_array_types[i]] + " " + \
                 work_array_names[i] + ";\n")
#
# Write out integer variables that will hold the size of each work
# array. This will be the same name as the work array, with an "l"
# in front of it.
#
  w1file.write("  int ")
  for i in range(len(work_array_names)):
    if i == (len(work_array_names)-1):
      w1file.write("l" + work_array_names[i] + ";\n")
    else:
      w1file.write("l" + work_array_names[i] + ", ")

#---------------------------------------------------------------------
#
# If any input variable has leftmost dimensions, include that variable
# here.
#
#---------------------------------------------------------------------
if have_leftmost:
  if isfunc:
    w1file.write("  int i, ndims_leftmost, size_leftmost, " + \
                  ret_arg.size_name + ";\n")
  else:
    w1file.write("  int i, ndims_leftmost, size_leftmost;\n")
else:
  if isfunc:  
    w1file.write("  int " + ret_arg.size_name + ";\n")
  
#---------------------------------------------------------------------
#
# Write the PyArg_ParseTuple code for retrieving each argument from the
# PyNGL function call.
#
#---------------------------------------------------------------------
w1file.write("""
/*
 * Retrieve arguments.
 */
""")

#
# Loop across input arguments and generate code that will
# retrieve them from the PyNGL script. You need to have an 'O' for
# every required argument in the list.
#
w1file.write('  if (!PyArg_ParseTuple(args,(char *)"')
for i in range(len(args)):
  w1file.write('O')
w1file.write(':' + pyngl_name + '",')

for i in range(len(args)):
  w1file.write('&' + args[i].arr_name)
  if i == (len(args)-1):
    w1file.write(')) {\n')
  else:
    w1file.write(',')

write_fatal_block(w1file,'argument parsing failed')

w1file.write("/*\n")
w1file.write(" * Start extracting array information.\n")
w1file.write(" */\n")

for i in range(len(args)):
  w1file.write("/*\n")
  w1file.write(" * Get argument # " + str(i) + "\n")
  w1file.write(" */\n")

  w1file.write("  arr = (PyArrayObject *) PyArray_ContiguousFromObject\n")
  w1file.write("                        (" + args[i].arr_name + \
               ",PyArray_DOUBLE,0,0);\n")

  w1file.write("  " + args[i].name + "        = (double *)arr->data;\n")
  w1file.write("  " + args[i].ndims_name + "  = arr->nd;\n")
  w1file.write("  " + args[i].dsizes_name + " = (int *)malloc(" + \
                      args[i].ndims_name + " * sizeof(int));\n")
  w1file.write("  for(i = 0; i < " + args[i].ndims_name + "; i++ ) " + \
                  args[i].dsizes_name + "[i] = arr->dimensions[i];\n")
#---------------------------------------------------------------------
#
# Write out code for doing some minimal error checking on input
# variable sizes.  We only need to check variable sizes that are 
# not hard-wired to a specific size (like "2 x 3 x 4" versus 
# "nx x ny x nz").
#
#
# Also, if a variable is to contain leftmost dimensions, then we need
# to make sure that it meets the bare minimum requirements for number
# of rightmost dimensions.
#
#---------------------------------------------------------------------
  if args[i].min_ndims > 0:
    w1file.write("""
/*
 * Check dimension sizes.
 */
""")
    w1file.write("  if(" + args[i].ndims_name + " < " + \
                str(args[i].min_ndims) + ") {\n")
    if (args[i].min_ndims == 1):
      write_fatal_block(w1file,'The ' + args[i].name + \
                   ' array must have at least ' + str(args[i].min_ndims) + \
                   ' dimension')
    else:
      write_fatal_block(w1file,'The ' + args[i].name + \
                   ' array must have at least ' + str(args[i].min_ndims) + \
                   ' dimensions')

  if not args[i].is_scalar:
    dstr = ""
    for j in range(len(args[i].dsizes_names)):
#
# If we're not dealing with a scalar, then write the code that 
# assigns the size of the rightmost dimensions, and then creates
# a variable that will contain the size of all the rightmost dimensions.
#
# For example, if the rightmost dimensions are nx, ny, and
# nz, then the code will be "nxnynz = nx * ny * nz". 
#
      if not args[i].dsizes_names[j] in global_dsizes_names_accum:
        if args[i].ndims > 0:
          if args[i].dsizes[j] == 0:
            w1file.write('  ' + args[i].dsizes_names[j] + ' = ' + \
                         args[i].dsizes_name + '[' + str(j) + '];\n')
          else:
            w1file.write('  ' + args[i].dsizes_names[j] + ' = ' + \
                         str(args[i].dsizes[j]) + ';\n')
        else:
          w1file.write('  ' + args[i].dsizes_names[j] + ' = ' + \
                      args[i].dsizes_name + '[' + args[i].ndims_name + '-' + 
                      str(int(args[i].min_ndims-j)) + '];\n')
        global_dsizes_names_accum.append(args[i].dsizes_names[j])
      else:
        if args[i].ndims > 0 and args[i].dsizes[j] == 0:
          w1file.write('  if(' + args[i].dsizes_name + '[' + str(j) + \
                       '] != ' + args[i].dsizes_names[j] + ') {\n')

          write_fatal_block(w1file,'The #' + str(j) + ' argument of ' + 
                            args[i].name + ' must be length ' + \
                            args[i].dsizes_names[j])

        elif args[i].ndims == 0:
          w1file.write('  if(' + args[i].dsizes_name + '[' + 
                      args[i].ndims_name + '-' + str(args[i].min_ndims-j) + \
                      '] != ' + args[i].dsizes_names[j] + ') {\n')
          write_fatal_block(w1file,'The ndims-' + \
                            str(int(args[i].min_ndims-j)) + \
                            ' dimension of ' + args[i].name + \
                            ' must be of length ' + \
                            args[i].dsizes_names[j])

      if args[i].min_ndims > 1 or args[i].ndims > 1:
        if j == 0:
          dstr = "  " + args[i].dsizes_names_str + " = " + \
                 args[i].dsizes_names[0]
        else:
          dstr = dstr + " * " + args[i].dsizes_names[j]

    if dstr != "":
      w1file.write(dstr + ";\n\n")

#---------------------------------------------------------------------
#
# Write out code to calculate size of leftmost dimensions, if any.
#
#---------------------------------------------------------------------
if have_leftmost:
  first    = True
  name_str = ""

  w1file.write("""
/*
 * Calculate size of leftmost dimensions.
 */
""")
  w1file.write("  size_leftmost  = 1;\n")

  for i in range(len(args)):
    if args[i].min_ndims > 0:
      if first:
        w1file.write("  ndims_leftmost = " + args[i].ndims_name + "-" + 
                     str(args[i].min_ndims) + ";\n");
        w1file.write("  for(i = 0; i < ndims_leftmost; i++) {\n")

        first_arg_name = args[i].name       # Keep track of this argument
        prev_arg_name  = first_arg_name
        first          = False
        second         = True
      else:
        if second:
          w1file.write("    if(" + args[i].dsizes_name + "[i] != dsizes_" + \
                       first_arg_name + "[i]")
          name_str      = prev_arg_name
          prev_arg_name = args[i].name
          second        = False
        else:
          name_str = name_str + ", " + prev_arg_name
          w1file.write(" ||\n       " + args[i].dsizes_name + \
                      "[i] != dsizes_" + first_arg_name + "[i]")
          prev_arg_name = args[i].name
#---------------------------------------------------------------------
#
# Write code to close leftmost dimensions loop.
#
#---------------------------------------------------------------------
  if name_str != "":
    w1file.write(") {\n")
    write_fatal_block(w1file,'The leftmost dimensions of ' + \
                      name_str + ' and ' + prev_arg_name + \
                     ' must be the same',spaces="    ")

  if not first:
    w1file.write("    size_leftmost *= dsizes_" + first_arg_name + "[i];\n")
    w1file.write("  }\n\n")

if isfunc:
#----------------------------------------------------------------------
# Code to calculate size of output array.
#----------------------------------------------------------------------
  w1file.write("""
/*
 * Calculate size of output array.
 */
""")

#
# Create string that will hold total dimension sizes of rightmost
# dimensions of return variable.
#
  if not ret_arg.is_scalar:
    if (ret_arg.min_ndims > 0) or (0 in ret_arg.dsizes):
      if ret_arg.min_ndims > 1 or ret_arg.ndims > 1:
        for j in range(len(ret_arg.dsizes_names)):
          if j == 0:
            ret_dstr = "  " + ret_arg.dsizes_names_str + " = " + \
                       ret_arg.dsizes_names[0]
          else:
            ret_dstr = ret_dstr + " * " + ret_arg.dsizes_names[j]

        w1file.write(ret_dstr + ";\n")

    w1file.write("  " + ret_arg.size_name + " = size_leftmost * " + \
                ret_arg.dsizes_names_str + ";\n")
  else:
    w1file.write("  " + ret_arg.size_name + " = ...need input here...;\n")

  w1file.write("""
/* 
 * Allocate space for output array.
 */
""")
  w1file.write("  " + ret_arg.name + " = (" + ret_arg.ctype + \
              " *)calloc(" + ret_arg.size_name + \
              ", sizeof(" + ret_arg.ctype + "));\n")

  w1file.write("  if(" + ret_arg.name + " == NULL) {\n")
  write_fatal_block(w1file,'Unable to allocate memory for output array')
#
# Set the missing value for the output, if any.
#
  if ret_has_missing and (num_input_has_missing > 0) and ret_msg_depend_input:
    w1file.write("  if(" + args[ret_msg_depend_index].has_msg_name + ") {\n")
    w1file.write("    if(" + ret_arg.type_name + " == NCL_double) " + \
                 ret_arg.msg_name + " = " + \
                 args[ret_msg_depend_index].msg_dname + ";\n")
    w1file.write("    else                 " + ret_arg.msg_name + " = " + \
                 args[ret_msg_depend_index].msg_fname + ";\n")
    w1file.write("    " + ret_arg.msg_dname + " = " + \
                 args[ret_msg_depend_index].msg_dname + ";\n")
    w1file.write("  }\n")

#
# If the dimension sizes are identical to one of the input variables'
# dimension sizes, then nothing is needed here.
#
  if not ret_size_depend_input:
    w1file.write("""
/* 
 * Allocate space for output dimension sizes and set them.
 */
""")
    if ret_arg.ndims > 0:
      w1file.write("  " + ret_arg.ndims_name + " = " + str(ret_arg.ndims) + \
                   ";\n")
    else:
      w1file.write("  " + ret_arg.ndims_name + " = ndims_leftmost + " + \
                  str(ret_arg.min_ndims) + ";\n")
    w1file.write("  " + ret_arg.dsizes_name + " = (int*)calloc(" + \
                ret_arg.ndims_name + ",sizeof(int));  \n")
    w1file.write("  if( " + ret_arg.dsizes_name + " == NULL ) {\n")
    write_fatal_block(w1file,'Unable to allocate memory for holding dimension sizes')

    if ret_arg.min_ndims > 0:
#
# Loop through input arguments until we find one that has leftmost
# dimensions, and then use its leftmost dimensions to assign 
# dimensions to the output array's dimension sizes array.
#
      for i in range(len(args)):
        if args[i].min_ndims > 0:
          w1file.write("  for(i = 0; i < " + ret_arg.ndims_name + "-" + \
                      str(ret_arg.min_ndims) + "; i++) " + \
                      ret_arg.dsizes_name + "[i] = " + args[i].dsizes_name + "[i];\n")
          break
      else:
        w1file.write("  for(i = 0; i < " + ret_arg.ndims_name + \
                    "; i++)" + ret_arg.dsizes_name + \
                    "[i] = ...need input here;\n")
      for i in range(ret_arg.min_ndims):
        w1file.write("  " + ret_arg.dsizes_name + "[" + ret_arg.ndims_name + 
                    "-" + str(ret_min_ndims-i) + "] = " + \
                    ret_dsizes_names[i] + ";\n")
    else:
      w1file.write("  for(i = 0; i < " + ret_arg.ndims_name + "; i++)" + \
                   ret_arg.dsizes_name + "[i] = ...need input here;\n")

#
# Write out code for the loop across leftmost dimensions (if any).
#
if have_leftmost:
  w1file.write("""
/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * subsection of the input arrays.
 */
""")
  index_str = "  "
  for i in range(len(index_names)):
    if i < (len(index_names)-1):
      index_str = index_str + index_names[i] + " = "
    else:
      index_str = index_str + index_names[i] + " = 0;\n"

  w1file.write(index_str)
  w1file.write("\n  for(i = 0; i < size_leftmost; i++) {\n")

#
# Code for pointing tmp_xxxx to appropriate location in x if it's
# a variable that has leftmost dimensions.
#
  for i in range(len(args)):
    if args[i].ndims == 0:
      w1file.write("    " + args[i].tmp_name + " = &((" + args[i].ctype + \
                   "*)" + args[i].name + ")[" + args[i].index_name + "];\n")

  for i in range(len(args)):
    name = args[i].name
#
# Write code for calling Fortran routine inside loop. The input variables
# and the output variable need to be pointed to appropriately.
#
  w1file.write("""
/*
 * Loop across leftmost dimensions and call the Fortran routine.
 */
""")
  w1file.write("    NGCALLF("+ lower(fortran_name) + "," + \
              upper(fortran_name) + ")(")
  for i in range(len(farg_names)):
    if i != 0:
      w1file.write(", ")
#
# Need to add code to check the ndims of farg_names so that the index
# thing is only added if you are dealing with a variable with variable
# dimensions (ndims=0).
#
    if farg_names[i] in input_var_names or farg_names[i] in ret_name:
      w1file.write("&" + farg_names[i] + "[index_" + farg_names[i] + "]")
    else:
      w1file.write(farg_cchar[i] + farg_names[i])

  w1file.write(");\n")

#
# Write out code for incrementing index variables, if any.
#
  for i in range(len(args)):
    if args[i].ndims == 0:
      w1file.write("    "  + args[i].index_name + " += " + \
                  args[i].dsizes_names_str + ";\n")
  if isfunc and ret_arg.ndims == 0:
    w1file.write("    "  + ret_arg.index_name + " += " + \
                ret_arg.dsizes_names_str + ";\n")

  w1file.write("  }\n")    # End "for" loop
else:
#
# We are dealing with a function that doesn't have any arguments
# with leftmost dimensions. This is unusual, but possible.
#
# Write code for calling Fortran routine not in a loop.
#
  w1file.write("""
/*
 * Call the Fortran routine.
 */
""")
  w1file.write("  NGCALLF("+ lower(fortran_name) + "," + \
              upper(fortran_name) + ")(")
  for i in range(len(farg_names)):
    if i != 0:
      w1file.write(", ")
    w1file.write(farg_cchar[i] + farg_names[i])

  w1file.write(");\n")

#
# Set up code for returning information back to PyNGL script.
#
if isfunc:
  w1file.write("""
/*
 * Return value back to PyNGL script.
 */
""")
#
# If the return can have a missing value, account for it here. Also,
# if the return can be float or double, then we have to return
# the correct missing value.
#
#  if ret_has_missing:
#    if ret_arg.ntype == "numeric":
#      w1file.write("  if(" + ret_arg.type_name + " != NCL_double) {\n")
#      w1file.write("    ret = NclReturnValue(" + ret_arg.name + "," + \
#                   ret_arg.ndims_name + "," + ret_arg.dsizes_name + \
#                   ",&" + ret_arg.msg_fname + "," + \
#                   ret_arg.type_name + ",0);\n")
#      w1file.write("  }\n")
#      w1file.write("  else {\n")
#      w1file.write("    ret = NclReturnValue(" + ret_arg.name + "," + \
#                   ret_arg.ndims_name + "," + ret_arg.dsizes_name + \
#                   ",&" + ret_arg.msg_dname + "," + \
#                   ret_arg.type_name + ",0);\n")
#      w1file.write("  }\n")
#    else:
#      w1file.write("  else {\n")
#      w1file.write("    ret = NclReturnValue(" + ret_arg.name + "," + \
#                   ret_arg.ndims_name + "," + ret_arg.dsizes_name + "," + \
#                   ret_arg.msg_name + "," + ret_arg.type_name + ",0);\n")
#      w1file.write("  }\n")
#  else:
# 
# Dummy "if" statement until we get the missing value stuff to work.
#
  if True:

    if ret_size_depend_input:
      w1file.write("  " + ret_arg.ret_name + \
                   " = (PyArrayObject *) PyArray_FromDimsAndData(" + \
                   args[ret_size_depend_index].ndims_name + "," + \
                   args[ret_size_depend_index].dsizes_name + \
                   ", PyArray_DOUBLE, (void *) " + ret_arg.name + ");\n")

    else:
      w1file.write("  " + ret_arg.ret_name + \
                   " = (PyArrayObject *) PyArray_FromDimsAndData(" + \
                   ret_arg.ndims_name + "," + ret_arg.dsizes_name + \
                   ", PyArray_DOUBLE, (void *) " + ret_arg.name + ");\n")

      w1file.write("  free(" + ret_arg.dsizes_name + ");\n")

  w1file.write("  PyArray_Return(" + ret_arg.ret_name + ");\n")
else:
  w1file.write("""
/*
 * This is a procedure, so no values are returned.
 */
""")
  wfile.write('    return Py_None;\n')

#---------------------------------------------------------------------
#
# Write out last line of PyNGL wrapper routine.
#
# }
#
#---------------------------------------------------------------------
w1file.write("}\n")

#
# Write the code for registering this function or procedure with PyNGL.
# This will be written to a different file, and needs to be copied
# to "wrapper.c".
#

w2file.write("extern void NGCALLF("+ lower(fortran_name) + "," +
             upper(fortran_name) + ")(")
for i in range(len(farg_types)):
  if i == 0:
    w2file.write(ctypes[farg_types[i]] + " *")
  else:
    w2file.write(", " + ctypes[farg_types[i]] + " *")
w2file.write(");\n\n")

w2file.write('#include "' + pyngl_nameP + '.c"\n')
w2file.write('{"' + pyngl_name + '", (PyCFunction)fplib_' + pyngl_name + \
             ', METH_VARARGS},\n')

#
# Close files.
#
w1file.close()
w2file.close()
