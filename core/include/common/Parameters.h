#pragma once

#include <QHash>
#include <QString>
#include <QVector>

class Parameters
{
public:

	// returns the singleton
	static Parameters* instance();

	// parses parameters in filename and overwrites it in the singleton
	static bool parse( QString filename );

	bool hasBool( QString name );
	bool getBool( QString name );
	void setBool( QString name, bool value );
	void toggleBool( QString name );

	bool hasInt( QString name );
	int getInt( QString name );
	void setInt( QString name, int value );

	bool hasIntArray( QString name );
	QVector< int > getIntArray( QString name );
	void setIntArray( QString name, QVector< int > values );

	bool hasFloat( QString name );
	float getFloat( QString name );
	void setFloat( QString name, float value );

	bool hasFloatArray( QString name );
	QVector< float > getFloatArray( QString name );
	void setFloatArray( QString name, QVector< float > values );

	bool hasString( QString name );
	QString getString( QString name );
	void setString( QString name, QString value );

	bool hasStringArray( QString name );
	QVector< QString > getStringArray( QString name );
	void setStringArray( QString name, QVector< QString > values );

private:

	Parameters();

	static Parameters* s_singleton;

	QHash< QString, bool > m_boolParameters;
	
	QHash< QString, int > m_intParameters;	
	QHash< QString, QVector< int > > m_intArrayParameters;

	QHash< QString, float > m_floatParameters;
	QHash< QString, QVector< float > > m_floatArrayParameters;

	QHash< QString, QString > m_stringParameters;
	QHash< QString, QVector< QString > > m_stringArrayParameters;

	friend QDataStream& operator << ( QDataStream& s, const Parameters& p );
	friend QDataStream& operator >> ( QDataStream& s, Parameters& p );
};

QDataStream& operator << ( QDataStream& s, const Parameters& p );
QDataStream& operator >> ( QDataStream& s, Parameters& p );
