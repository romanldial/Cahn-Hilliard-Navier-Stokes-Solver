set(_EXAMPLES
  1dheat
  2dHeatEx1
  ch_couple_ex1
  chem_pot_ex1
  heat
  heat2-LILS
  heat2EA
  heat3-LILS
)

foreach(_ex IN LISTS _EXAMPLES)
  # CMake target names cannot contain hyphens; map them to underscores.
  string(MAKE_C_IDENTIFIER "${_ex}" _target)
  add_executable("${_target}" "${_ex}.cpp")
  set_target_properties("${_target}" PROPERTIES OUTPUT_NAME "${_ex}")
  target_link_libraries("${_target}" PRIVATE chns)
endforeach()
