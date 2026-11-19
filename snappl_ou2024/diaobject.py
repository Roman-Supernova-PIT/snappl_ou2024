__all__ = [ 'DiaObjectOU2024' ]

from snappl.snappl_http import retry_post
from snappl.config import Config
from snappl.logger import SNLogger
from snappl.diaobject import DiaObject


class DiaObjectOU2024( DiaObject ):
    """A transient from the OpenUniverse 2024 sims."""

    def __init__( self, *args, **kwargs ):
        """Don't call this constructor directly.  Use DiaObject.find_objects."""
        super().__init__( *args, **kwargs )

        # Non-standard fields
        self.host_id = None
        self.gentype = None
        self.model_name = None
        self.start_mjd = None
        self.end_mjd = None
        self.z_cmb = None
        self.mw_ebv = None
        self.mw_extinction_applied = None
        self.av = None
        self.rv = None
        self.v_pec = None
        self.host_ra = None
        self.host_dec = None
        self.host_mag_g = None
        self.host_mag_i = None
        self.host_mag_f = None
        self.host_sn_sep = None
        self.peak_mag_g = None
        self.peak_mag_i = None
        self.peak_mag_f = None
        self.lens_dmu = None
        self.lens_dmu_applied = None
        self.model_params = None

    @classmethod
    def _find_objects( cls, subset=None,
                       name=None,
                       ra=None,
                       dec=None,
                       radius=1.0,
                       mjd_peak_min=None,
                       mjd_peak_max=None,
                       mjd_discovery_min=None,
                       mjd_discovery_max=None,
                       mjd_start_min=None,
                       mjd_start_max=None,
                       mjd_end_min=None,
                       mjd_end_max=None,
                       diaobject_id=None,
                      ):
        if any( i is not None for i in [ mjd_peak_min, mjd_peak_max, mjd_discovery_min, mjd_discovery_max ] ):
            raise NotImplementedError( "DiaObjectOU2024 doesn't support searching on mjd_peak or mjd_discovery" )

        if diaobject_id is not None:
            SNLogger.warning("DiaObject OU2024 ignoring diaobject_id parameter in find_objects")

        params = {}

        if ( ra is None ) != ( dec is None ):
            raise ValueError( "Pass both or neither of ra/dec, not just one." )

        if ra is not None:
            if radius is None:
                raise ValueError( "ra/dec requires a radius" )
            params['ra'] = float( ra )
            params['dec'] = float( dec )
            params['radius'] = float( radius )

        if name is not None:
            params['id'] = int( name )

        if mjd_start_min is not None:
            params['mjd_start_min'] = float( mjd_start_min )

        if mjd_start_max is not None:
            params['mjd_start_max'] = float( mjd_start_max )

        if mjd_end_min is not None:
            params['mjd_end_min'] = float( mjd_end_min )

        if mjd_end_min is not None:
            params['mjd_end_max'] = float( mjd_end_max )

        simdex = Config.get().value( 'system.ou24.simdex_server' )
        res = retry_post( f'{simdex}/findtransients', json=params )
        objinfo = res.json()

        diaobjects = []
        for i in range( len( objinfo['id'] ) ):
            props = { prop: objinfo[prop][i] for prop in
                      [ 'healpix', 'host_id', 'gentype', 'model_name', 'z_cmb', 'mw_ebv', 'mw_extinction_applied',
                        'av', 'rv', 'v_pec', 'host_ra', 'host_dec', 'host_mag_g', 'host_mag_i', 'host_mag_f',
                        'host_sn_sep', 'peak_mag_g', 'peak_mag_i', 'peak_mag_f', 'lens_dmu',
                        'lens_dmu_applied', 'model_params' ] }

            diaobj = DiaObjectOU2024( name=str( objinfo['id'][i] ),
                                      ra=objinfo['ra'][i],
                                      dec=objinfo['dec'][i],
                                      mjd_peak=objinfo['peak_mjd'][i],
                                      mjd_start=objinfo['start_mjd'][i],
                                      mjd_end=objinfo['end_mjd'][i],
                                      properties=props )
            diaobjects.append( diaobj )

        return diaobjects
