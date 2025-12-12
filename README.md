# failure-prediction
The objective of this procedure is to evaluate the sub-assemblies before proceeding to the actual finite element analysis or simulation. Pre-engineering is done by finding the stress & deflection of the railing sub-assembly (here test & unknown data values), which assist engineering to verify material selection & its strength by using earlier simulation or tested data based on features like part’s thickness & overall sizes (here training dats).

The first linear regression machine learning model gets trained on earlier data. This model can then predict new values for given new inputs according to earlier learned data. 
