# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class SearchTrain(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)  # Field name made lowercase.
    action = models.CharField(db_column='action', max_length=255)
    input = models.CharField(db_column='input', max_length=255)

    class Meta:
        managed = False
        db_table = 'search_train'


class CommentLikes(models.Model):
    id = models.AutoField(db_column='Id', primary_key=True)  # Field name made lowercase.
    idowner = models.ForeignKey('Customers', models.DO_NOTHING, db_column='IdOwner')  # Field name made lowercase.
    idcomment = models.ForeignKey('Comments', models.DO_NOTHING, db_column='IdComment')  # Field name made lowercase.
    status = models.CharField(db_column='Status', max_length=255)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'comment_likes'


class CommentPictures(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True)  # Field name made lowercase.
    bgcolor = models.CharField(db_column='BgColor', max_length=10, blank=True, null=True)  # Field name made lowercase.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    height = models.IntegerField(db_column='Height', blank=True, null=True)  # Field name made lowercase.
    photodetailurl = models.CharField(db_column='PhotoDetailUrl', max_length=255, blank=True, null=True)  # Field name made lowercase.
    totallikes = models.IntegerField(db_column='TotalLikes', blank=True, null=True)  # Field name made lowercase.
    url = models.CharField(db_column='Url', max_length=255, blank=True, null=True)  # Field name made lowercase.
    width = models.IntegerField(db_column='Width', blank=True, null=True)  # Field name made lowercase.
    commentid = models.ForeignKey('Comments', models.DO_NOTHING, db_column='CommentId', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'comment_pictures'



class Comments(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True)  # Field name made lowercase.
    avgrating = models.FloatField(db_column='AvgRating', blank=True, null=True)  # Field name made lowercase.
    createddate = models.CharField(db_column='CreatedDate', max_length=255, blank=True, null=True)  # Field name made lowercase.
    createdontimediff = models.CharField(db_column='CreatedOnTimeDiff', max_length=255, blank=True, null=True)  # Field name made lowercase.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    devicename = models.CharField(db_column='DeviceName', max_length=255, blank=True, null=True)  # Field name made lowercase.
    devicetype = models.IntegerField(db_column='DeviceType', blank=True, null=True)  # Field name made lowercase.
    deviceurl = models.CharField(db_column='DeviceUrl', max_length=255, blank=True, null=True)  # Field name made lowercase.
    isfirstuserreview = models.IntegerField(db_column='IsFirstUserReview', blank=True, null=True)  # Field name made lowercase.
    isliked = models.IntegerField(db_column='IsLiked', blank=True, null=True)  # Field name made lowercase.
    owner = models.ForeignKey('Customers', models.DO_NOTHING, db_column='Owner_id', blank=True, null=True)  # Field name made lowercase.
    resid = models.ForeignKey('Restaurants', models.DO_NOTHING, db_column='ResId', blank=True, null=True)  # Field name made lowercase.
    title = models.TextField(db_column='Title', blank=True, null=True)  # Field name made lowercase.
    totallike = models.IntegerField(db_column='TotalLike', blank=True, null=True)  # Field name made lowercase.
    totalpictures = models.IntegerField(db_column='TotalPictures', blank=True, null=True)  # Field name made lowercase.
    totalview = models.IntegerField(db_column='TotalView', blank=True, null=True)  # Field name made lowercase.
    typename = models.CharField(db_column='TypeName', max_length=255, blank=True, null=True)  # Field name made lowercase.
    url = models.CharField(db_column='Url', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'comments'



class Customers(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True)  # Field name made lowercase.
    avatar = models.CharField(db_column='Avatar', max_length=255, blank=True, null=True)  # Field name made lowercase.
    displayname = models.TextField(db_column='DisplayName', blank=True, null=True)  # Field name made lowercase.
    isfollow = models.IntegerField(db_column='IsFollow', blank=True, null=True)  # Field name made lowercase.
    isverified = models.IntegerField(db_column='IsVerified', blank=True, null=True)  # Field name made lowercase.
    level = models.CharField(db_column='Level', max_length=20, blank=True, null=True)  # Field name made lowercase.
    rank = models.IntegerField(db_column='Rank', blank=True, null=True)  # Field name made lowercase.
    totalpictures = models.IntegerField(db_column='TotalPictures', blank=True, null=True)  # Field name made lowercase.
    totalreviews = models.IntegerField(db_column='TotalReviews', blank=True, null=True)  # Field name made lowercase.
    url = models.CharField(db_column='Url', max_length=255, blank=True, null=True)  # Field name made lowercase.
    urlalbums = models.CharField(db_column='UrlAlbums', max_length=255, blank=True, null=True)  # Field name made lowercase.
    urlreviews = models.CharField(db_column='UrlReviews', max_length=255, blank=True, null=True)  # Field name made lowercase.
    username = models.CharField(db_column='Username', max_length=255, blank=True, null=True)  # Field name made lowercase.
    verifyingpercent = models.IntegerField(db_column='VerifyingPercent', blank=True, null=True)  # Field name made lowercase.
    status = models.CharField(db_column='Status', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'customers'







class Restaurants(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True)  # Field name made lowercase.
    address = models.CharField(db_column='Address', max_length=255, blank=True, null=True)  # Field name made lowercase.
    avgrating = models.IntegerField(db_column='AvgRating', blank=True, null=True)  # Field name made lowercase.
    description = models.CharField(db_column='Description', max_length=255, blank=True, null=True)  # Field name made lowercase.
    distance = models.FloatField(db_column='Distance', blank=True, null=True)  # Field name made lowercase.
    haspromotion = models.IntegerField(db_column='HasPromotion', blank=True, null=True)  # Field name made lowercase.
    isopening = models.IntegerField(db_column='IsOpening', blank=True, null=True)  # Field name made lowercase.
    latitude = models.FloatField(db_column='Latitude', blank=True, null=True)  # Field name made lowercase.
    longitude = models.FloatField(db_column='Longitude', blank=True, null=True)  # Field name made lowercase.
    locationurlrewritename = models.CharField(db_column='LocationUrlRewriteName', max_length=255, blank=True, null=True)  # Field name made lowercase.
    name = models.CharField(db_column='Name', max_length=255, blank=True, null=True)  # Field name made lowercase.
    rescreatedon = models.CharField(db_column='ResCreatedOn', max_length=255, blank=True, null=True)  # Field name made lowercase.
    photourl = models.CharField(db_column='PhotoUrl', max_length=255, blank=True, null=True)  # Field name made lowercase.
    pricemax = models.IntegerField(db_column='PriceMax', blank=True, null=True)  # Field name made lowercase.
    pricemaxdisplay = models.CharField(db_column='PriceMaxDisplay', max_length=10, blank=True, null=True)  # Field name made lowercase.
    pricemin = models.IntegerField(db_column='PriceMin', blank=True, null=True)  # Field name made lowercase.
    pricemindisplay = models.CharField(db_column='PriceMinDisplay', max_length=10, blank=True, null=True)  # Field name made lowercase.
    promotionid = models.IntegerField(db_column='PromotionId', blank=True, null=True)  # Field name made lowercase.
    promotiontitle = models.CharField(db_column='PromotionTitle', max_length=255, blank=True, null=True)  # Field name made lowercase.
    promotionurl = models.CharField(db_column='PromotionUrl', max_length=255, blank=True, null=True)  # Field name made lowercase.
    resurlalbums = models.CharField(db_column='ResUrlAlbums', max_length=255, blank=True, null=True)  # Field name made lowercase.
    resurlreviews = models.CharField(db_column='ResUrlReviews', max_length=255, blank=True, null=True)  # Field name made lowercase.
    status = models.IntegerField(db_column='Status', blank=True, null=True)  # Field name made lowercase.
    restaurantstatus = models.IntegerField(db_column='RestaurantStatus', blank=True, null=True)  # Field name made lowercase.
    totalcheckins = models.IntegerField(db_column='TotalCheckIns', blank=True, null=True)  # Field name made lowercase.
    totalfavourites = models.IntegerField(db_column='TotalFavourites', blank=True, null=True)  # Field name made lowercase.
    totalpictures = models.IntegerField(db_column='TotalPictures', blank=True, null=True)  # Field name made lowercase.
    totalreviews = models.IntegerField(db_column='TotalReviews', blank=True, null=True)  # Field name made lowercase.
    totalsaves = models.IntegerField(db_column='TotalSaves', blank=True, null=True)  # Field name made lowercase.
    totalservice = models.IntegerField(db_column='TotalService', blank=True, null=True)  # Field name made lowercase.
    url = models.CharField(db_column='Url', max_length=255, blank=True, null=True)  # Field name made lowercase.
    urlrewritename = models.CharField(db_column='UrlRewriteName', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'restaurants'


class Similarity(models.Model):
    created = models.DateField()
    source = models.CharField(max_length=16, db_index=True)
    target = models.CharField(max_length=16)
    similarity = models.DecimalField(max_digits=8, decimal_places=7)

    class Meta:
        db_table = 'similarity'

    def __str__(self):
        return "[({} => {}) sim = {}]".format(self.source,
                                              self.target,
                                              self.similarity)
