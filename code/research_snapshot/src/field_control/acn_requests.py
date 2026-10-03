"""Query-time user declarations, distinct from realized session outcomes."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re

REQUEST_FIELDS=('kWhRequested','modifiedAt','requestedDeparture')


def timestamp(value):
    if not value:
        return None
    try:
        value=str(value)
        parsed=parsedate_to_datetime(value) if ',' in value else datetime.fromisoformat(value.replace('Z','+00:00'))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def request_asof(revisions, query):
    """Return the latest timestamped declaration known by query time."""
    query=timestamp(query)
    if query is None:
        raise ValueError('Invalid query time')
    eligible=[]
    for revision in revisions or []:
        when=timestamp(revision.get('modifiedAt'))
        if when is not None and when<=query:
            eligible.append((when,revision))
    if not eligible:
        return None
    selected=max(eligible,key=lambda item:item[0])[1]
    return {key:selected[key] for key in REQUEST_FIELDS if key in selected}


def static_filename(session_id):
    """Match the author static archive's encoding of the API session ID."""
    fields=str(session_id).split('_',4)
    if len(fields)!=5 or not all(x.isdigit() for x in fields[:4]):
        raise ValueError('Unrecognized session identifier')
    time=timestamp(fields[4])
    if time is None:
        raise ValueError('Session identifier has no timestamp')
    return '-'.join(fields[:4])+'-'+time.strftime('%Y-%m-%dT%H-%M-%S-%f')+'.csv.gz'


def declaration_features(requested_kwh, departure_hours, delivered_kwh):
    """Declared quantities and masks; optional outstanding-request proxy.

    The proxy is not true battery state of charge or realized remaining demand.
    """
    import numpy as np
    requested=np.asarray(requested_kwh,dtype=float)
    departure=np.asarray(departure_hours,dtype=float)
    delivered=np.asarray(delivered_kwh,dtype=float)
    known=np.isfinite(requested)&(requested>=0)
    deadline=np.isfinite(departure)
    gap_known=known&np.isfinite(delivered)&(delivered>=0)
    raw=np.column_stack([np.where(known,requested,0),np.where(deadline,departure,0),known,deadline]).astype(np.float32)
    gap=np.column_stack([np.where(gap_known,np.maximum(requested-delivered,0),0),gap_known]).astype(np.float32)
    return raw,gap


DECLARATION_NAMES=('requested_kwh','requested_departure_hours','request_observed',
                   'departure_observed','outstanding_requested_kwh','outstanding_observed')


def declaration_stats(values,train):
    import numpy as np
    mean=np.zeros(6,dtype=np.float32);scale=np.ones(6,dtype=np.float32)
    for value,mask in [(0,2),(1,3),(4,5)]:
        x=values[train,value][values[train,mask]>0]
        if len(x):mean[value]=x.mean();scale[value]=max(float(x.std()),1e-6)
    return mean,scale


def normalized_declarations(values,stats,view):
    import numpy as np
    if view not in ['padded','raw','gap']:raise ValueError('Unknown declaration view')
    result=np.array((values-stats[0])/stats[1],dtype=np.float32,copy=True)
    for value,mask in [(0,2),(1,3),(4,5)]:result[values[:,mask]==0,value]=0
    if view=='padded':result[:]=0
    if view=='raw':result[:,4:]=0
    return result
